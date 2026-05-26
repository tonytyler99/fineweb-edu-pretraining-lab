"""IterableDataset over uint16 shards for training and validation."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from src.data.shards import list_shards, load_shard


class ShardDataset(IterableDataset):
    """Sample fixed-length ``(x, y)`` windows from uint16 token shards.

    Shards are memmapped on demand; each iteration step picks a uniformly
    random shard, then a uniformly random offset within it. Infinite stream —
    the training loop controls how many batches to consume.
    """

    def __init__(
        self,
        shard_dir: Path,
        split: str,
        block_size: int,
        seed: int = 0,
    ) -> None:
        """Open all shards matching ``{split}_*.bin`` under ``shard_dir``.

        Args:
            shard_dir: Directory containing the prepared shards.
            split: Filename prefix, ``"train"`` or ``"val"``.
            block_size: Context length ``T``. Each sample is ``T+1`` tokens
                so that ``y`` can be ``x`` shifted by one.
            seed: Base RNG seed; worker id is added when ``num_workers > 0``.
        """
        self._shard_paths = list_shards(shard_dir, split)
        if not self._shard_paths:
            raise FileNotFoundError(
                f"no shards found for split={split!r} under {shard_dir}"
            )
        self._block_size = block_size
        self._seed = seed

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        info = get_worker_info()
        # Per-worker seed offset keeps shuffling independent across workers
        # without needing a shared RNG. With num_workers=0 (Windows default)
        # info is None and we just use the base seed.
        seed = self._seed + (info.id if info is not None else 0)
        rng = np.random.default_rng(seed)
        block = self._block_size
        shards = self._shard_paths

        while True:
            shard = load_shard(shards[rng.integers(0, len(shards))])
            # +1 because y is x shifted by one token; need block+1 contiguous tokens.
            start = int(rng.integers(0, shard.size - block - 1))
            window = shard[start : start + block + 1]
            # torch lacks a uint16 dtype, so cast to int64 (the embedding dtype).
            x = torch.from_numpy(window[:-1].astype(np.int64))
            y = torch.from_numpy(window[1:].astype(np.int64))
            yield x, y
