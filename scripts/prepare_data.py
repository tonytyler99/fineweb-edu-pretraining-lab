"""Orchestrate FineWeb-Edu download, tokenization, and shard writing."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from src.data.download import iter_fineweb_edu_docs
from src.data.shards import (
    BYTES_PER_TOKEN,
    ShardWriter,
    list_shards,
)
from src.data.tokenize import iter_encoded

REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_FILENAME = "prep_state.json"


@dataclass(frozen=True)
class PrepConfig:
    """Resolved data-prep configuration (YAML merged with CLI overrides)."""

    repo: str
    subset: str
    hf_split: str
    shard_dir: Path
    encoding: str
    tokens_per_shard: int
    val_tokens: int
    max_docs: int | None
    seed: int


@dataclass(frozen=True)
class PrepState:
    """Checkpoint metadata persisted to ``prep_state.json``."""

    docs_consumed: int
    tokens_written: int
    val_complete: bool
    train_shards_complete: int


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_config(yaml_cfg: dict[str, Any], args: argparse.Namespace) -> PrepConfig:
    shard_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / yaml_cfg["paths"]["shard_dir"]
    max_docs = args.max_docs if args.max_docs is not None else yaml_cfg["prep"]["max_docs"]
    return PrepConfig(
        repo=yaml_cfg["dataset"]["repo"],
        subset=yaml_cfg["dataset"]["subset"],
        hf_split=yaml_cfg["dataset"]["split"],
        shard_dir=shard_dir,
        encoding=yaml_cfg["tokenization"]["encoding"],
        tokens_per_shard=int(yaml_cfg["shards"]["tokens_per_shard"]),
        val_tokens=int(yaml_cfg["shards"]["val_tokens"]),
        max_docs=max_docs,
        seed=int(yaml_cfg["prep"]["seed"]),
    )


def _load_state(state_path: Path) -> PrepState:
    if not state_path.exists():
        return PrepState(0, 0, False, 0)
    return PrepState(**json.loads(state_path.read_text(encoding="utf-8")))


def _save_state(state_path: Path, state: PrepState) -> None:
    state_path.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")


def _wipe_outputs(out_dir: Path) -> None:
    """Delete every shard and the state file under ``out_dir``."""
    if not out_dir.exists():
        return
    for p in out_dir.glob("*_*.bin"):
        p.unlink()
    state_path = out_dir / STATE_FILENAME
    if state_path.exists():
        state_path.unlink()


def _drop_partial_trailing(out_dir: Path, split: str, expected_tokens: int) -> None:
    """Remove the last shard of ``split`` if its size != ``expected_tokens * 2``."""
    shards = list_shards(out_dir, split)
    if not shards:
        return
    last = shards[-1]
    if last.stat().st_size != expected_tokens * BYTES_PER_TOKEN:
        last.unlink()


def _ensure_consistent(out_dir: Path, state: PrepState, cfg: PrepConfig) -> None:
    """Match the on-disk shard set to the state checkpoint."""
    # State is only saved at shard-close boundaries, so an unrecorded partial
    # shard on disk must come from an interrupted run — delete it.
    if state.val_complete:
        _drop_partial_trailing(out_dir, "val", cfg.val_tokens)
    else:
        for p in list_shards(out_dir, "val"):
            p.unlink()
    _drop_partial_trailing(out_dir, "train", cfg.tokens_per_shard)


def _dry_run(cfg: PrepConfig) -> None:
    """Tokenize a small slice and print summary stats; write nothing."""
    cap = cfg.max_docs if cfg.max_docs is not None else 1000
    docs = iter_fineweb_edu_docs(subset=cfg.subset, split=cfg.hf_split, max_docs=cap)
    sizes: list[int] = []
    for tokens in tqdm(iter_encoded(docs), total=cap, desc="dry-run", unit="doc"):
        sizes.append(int(tokens.size))
    n = len(sizes)
    total = sum(sizes)
    print(f"\nDry-run over {n:,} docs:")
    print(f"  total tokens : {total:,}")
    print(f"  mean / doc   : {total / max(n, 1):.1f}")
    print(f"  min / max    : {min(sizes)} / {max(sizes)}")
    print(f"  first 5 sizes: {sizes[:5]}")


def _run_prep(cfg: PrepConfig) -> None:
    out_dir = cfg.shard_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / STATE_FILENAME
    state = _load_state(state_path)
    _ensure_consistent(out_dir, state, cfg)

    val_complete = state.val_complete
    train_shards_done = state.train_shards_complete
    docs_done = state.docs_consumed
    tokens_done = state.tokens_written

    docs_iter = iter_fineweb_edu_docs(subset=cfg.subset, split=cfg.hf_split, max_docs=cfg.max_docs)
    if docs_done > 0:
        print(f"Resuming after {docs_done:,} docs / {tokens_done:,} tokens.")
        docs_iter = itertools.islice(docs_iter, docs_done, None)

    val_writer = (
        None if val_complete else ShardWriter(out_dir, "val", tokens_per_shard=cfg.val_tokens)
    )
    train_writer = ShardWriter(
        out_dir,
        "train",
        tokens_per_shard=cfg.tokens_per_shard,
        start_index=train_shards_done,
    )

    tokens_to_val = 0 if val_complete else cfg.val_tokens - tokens_done

    progress = tqdm(
        iter_encoded(docs_iter),
        total=cfg.max_docs,
        initial=docs_done,
        desc="tokenize",
        unit="doc",
        smoothing=0.05,
    )

    def _checkpoint() -> None:
        _save_state(
            state_path,
            PrepState(
                docs_consumed=docs_done,
                tokens_written=tokens_done,
                val_complete=val_complete,
                train_shards_complete=train_shards_done,
            ),
        )

    try:
        for tokens in progress:
            n = int(tokens.size)
            if val_writer is not None and tokens_to_val > 0:
                if n <= tokens_to_val:
                    val_writer.write(tokens)
                    tokens_to_val -= n
                else:
                    val_writer.write(tokens[:tokens_to_val])
                    train_writer.write(tokens[tokens_to_val:])
                    tokens_to_val = 0
                if tokens_to_val == 0:
                    val_writer.close()
                    val_writer = None
                    val_complete = True
                    docs_done += 1
                    tokens_done += n
                    _checkpoint()
                    continue
            else:
                train_writer.write(tokens)

            docs_done += 1
            tokens_done += n

            # Train shard completion is implied when token count crosses a shard
            # boundary; save once per crossing so resume granularity = 1 shard.
            if val_complete:
                new_done = (tokens_done - cfg.val_tokens) // cfg.tokens_per_shard
                if new_done > train_shards_done:
                    train_shards_done = new_done
                    _checkpoint()
    finally:
        if val_writer is not None:
            val_writer.close()
        train_writer.close()
        _checkpoint()

    val_shards = list_shards(out_dir, "val")
    train_shards = list_shards(out_dir, "train")
    print(f"\nDone. {docs_done:,} docs / {tokens_done:,} tokens written.")
    print(f"  val   : {len(val_shards)} shard(s)")
    print(f"  train : {len(train_shards)} shard(s)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare FineWeb-Edu pretraining shards.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "data" / "fineweb_edu_10bt.yaml",
        help="Path to data-prep YAML config.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Cap on consumed docs; overrides config (use for dry-runs).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override the shard output directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing shards and state before running.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Tokenize a sample, print stats, write nothing.",
    )
    args = parser.parse_args(argv)

    yaml_cfg = _load_yaml(args.config)
    cfg = _resolve_config(yaml_cfg, args)

    if args.overwrite:
        _wipe_outputs(cfg.shard_dir)
        print(f"Wiped existing shards in {cfg.shard_dir}.")

    if args.dry_run:
        _dry_run(cfg)
        return

    _run_prep(cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
