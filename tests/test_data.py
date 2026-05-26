"""Unit tests for the data pipeline. No network, no GPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tiktoken
import torch

from src.data.loader import ShardDataset
from src.data.shards import (
    BYTES_PER_TOKEN,
    ShardWriter,
    list_shards,
    load_shard,
    shard_path,
)
from src.data.tokenize import encode_doc, iter_encoded

# Pulled from the encoder rather than re-importing a constant so the test
# stays valid if the production module stops exporting it (current state).
_EOT_ID = tiktoken.get_encoding("gpt2").eot_token

# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------


def test_encode_doc_roundtrips_text_minus_eot() -> None:
    text = "The quick brown fox jumps over the lazy dog."
    enc = tiktoken.get_encoding("gpt2")
    tokens = encode_doc(text)
    assert enc.decode(tokens[:-1].tolist()) == text


def test_encode_doc_appends_eot() -> None:
    tokens = encode_doc("hello world")
    assert int(tokens[-1]) == _EOT_ID


def test_encode_doc_empty_text_is_eot_only() -> None:
    tokens = encode_doc("")
    assert tokens.shape == (1,)
    assert int(tokens[0]) == _EOT_ID


def test_encode_doc_returns_uint16() -> None:
    assert encode_doc("a").dtype == np.uint16


def test_iter_encoded_yields_one_array_per_doc() -> None:
    arrays = list(iter_encoded(["foo", "bar", "baz"]))
    assert len(arrays) == 3
    for arr in arrays:
        assert arr.dtype == np.uint16
        assert int(arr[-1]) == _EOT_ID


# ---------------------------------------------------------------------------
# shards
# ---------------------------------------------------------------------------


def test_shard_writer_roundtrip(tmp_path: Path) -> None:
    tokens = np.arange(500, dtype=np.uint16)
    with ShardWriter(tmp_path, "train", tokens_per_shard=1000) as w:
        w.write(tokens)
    [path] = list_shards(tmp_path, "train")
    np.testing.assert_array_equal(load_shard(path), tokens)


def test_shard_writer_rotates_at_capacity(tmp_path: Path) -> None:
    tokens = np.arange(250, dtype=np.uint16)
    with ShardWriter(tmp_path, "train", tokens_per_shard=100) as w:
        w.write(tokens)
    sizes = [p.stat().st_size // BYTES_PER_TOKEN for p in list_shards(tmp_path, "train")]
    assert sizes == [100, 100, 50]


def test_shard_writer_handles_many_small_writes(tmp_path: Path) -> None:
    with ShardWriter(tmp_path, "train", tokens_per_shard=100) as w:
        for _ in range(5):
            w.write(np.full(50, 7, dtype=np.uint16))
    sizes = [p.stat().st_size // BYTES_PER_TOKEN for p in list_shards(tmp_path, "train")]
    assert sizes == [100, 100, 50]


def test_shard_writer_rejects_wrong_dtype(tmp_path: Path) -> None:
    bad = np.arange(10, dtype=np.int32)
    with ShardWriter(tmp_path, "train") as w, pytest.raises(ValueError):
        w.write(bad)


def test_shard_writer_close_is_idempotent(tmp_path: Path) -> None:
    w = ShardWriter(tmp_path, "train", tokens_per_shard=100)
    w.write(np.arange(10, dtype=np.uint16))
    w.close()
    w.close()  # must not raise
    assert len(list_shards(tmp_path, "train")) == 1


def test_shard_writer_start_index_continues_numbering(tmp_path: Path) -> None:
    with ShardWriter(tmp_path, "train", tokens_per_shard=10) as w:
        w.write(np.zeros(20, dtype=np.uint16))
    with ShardWriter(tmp_path, "train", tokens_per_shard=10, start_index=2) as w:
        w.write(np.ones(20, dtype=np.uint16))
    assert [p.name for p in list_shards(tmp_path, "train")] == [
        "train_000000.bin",
        "train_000001.bin",
        "train_000002.bin",
        "train_000003.bin",
    ]


def test_list_shards_returns_sorted_paths(tmp_path: Path) -> None:
    for idx in (3, 0, 2, 1):
        shard_path(tmp_path, "train", idx).write_bytes(b"\x00\x00")
    assert [p.name for p in list_shards(tmp_path, "train")] == [
        "train_000000.bin",
        "train_000001.bin",
        "train_000002.bin",
        "train_000003.bin",
    ]


# ---------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------


def test_shard_dataset_yields_shifted_pair(tmp_path: Path) -> None:
    # Increasing values make ``y[i] == x[i] + 1`` trivially verifiable.
    tokens = np.arange(1000, dtype=np.uint16)
    with ShardWriter(tmp_path, "train", tokens_per_shard=2000) as w:
        w.write(tokens)

    block = 64
    ds = ShardDataset(tmp_path, "train", block_size=block, seed=0)
    it = iter(ds)
    for _ in range(20):
        x, y = next(it)
        assert x.shape == (block,)
        assert y.shape == (block,)
        assert x.dtype == torch.int64
        assert torch.equal(x[1:], y[:-1])


def test_shard_dataset_missing_split_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ShardDataset(tmp_path, "train", block_size=8)
