"""uint16 binary shard writer and reader for tokenized pretraining data."""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Final

import numpy as np

SHARD_DTYPE: Final = np.uint16
BYTES_PER_TOKEN: Final = 2
DEFAULT_TOKENS_PER_SHARD: Final = 100_000_000  # ~200 MB per file
_WRITE_BUFFER_BYTES: Final = 1 * 1024 * 1024  # 1 MB cuts per-write syscall cost


def shard_path(out_dir: Path, split: str, index: int) -> Path:
    """Return the canonical path ``{out_dir}/{split}_{index:06d}.bin``."""
    return out_dir / f"{split}_{index:06d}.bin"


def list_shards(out_dir: Path, split: str) -> list[Path]:
    """List shards for ``split`` in deterministic name order."""
    return sorted(out_dir.glob(f"{split}_*.bin"))


def load_shard(path: Path) -> np.memmap:
    """Open a shard as a read-only ``uint16`` memmap.

    Returns:
        1-D ``numpy.memmap``. Lazily paged in by the OS, safe for random
        offset sampling without loading the file into RAM.
    """
    return np.memmap(path, dtype=SHARD_DTYPE, mode="r")


class ShardWriter:
    """Append ``uint16`` tokens to rotating fixed-size shard files."""

    def __init__(
        self,
        out_dir: Path,
        split: str,
        tokens_per_shard: int = DEFAULT_TOKENS_PER_SHARD,
        start_index: int = 0,
    ) -> None:
        """Open a writer rooted at ``out_dir``.

        Args:
            out_dir: Directory to create shards in. Created if missing.
            split: Filename prefix, e.g. ``"train"`` or ``"val"``.
            tokens_per_shard: Token count that triggers a roll to the next file.
            start_index: First shard index. Set >0 to resume past existing shards.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        self._out_dir = out_dir
        self._split = split
        self._tokens_per_shard = tokens_per_shard
        self._next_index = start_index
        self._current_file: BinaryIO | None = None
        self._tokens_in_current = 0

    def write(self, tokens: np.ndarray) -> None:
        """Append a 1-D ``uint16`` array, rolling shards as they fill.

        Args:
            tokens: 1-D ``uint16`` array. Other dtypes raise ``ValueError``.
        """
        if tokens.dtype != SHARD_DTYPE:
            raise ValueError(f"expected dtype {SHARD_DTYPE}, got {tokens.dtype}")
        # tokens may straddle multiple shards; loop until consumed.
        while tokens.size > 0:
            if self._current_file is None:
                self._open_next_shard()
            assert self._current_file is not None  # for type-checker
            room = self._tokens_per_shard - self._tokens_in_current
            chunk = tokens[:room]
            self._current_file.write(chunk.tobytes())
            self._tokens_in_current += chunk.size
            tokens = tokens[room:]
            if self._tokens_in_current >= self._tokens_per_shard:
                self._close_current()

    def close(self) -> None:
        """Flush and close the current shard. Safe to call multiple times."""
        self._close_current()

    def _open_next_shard(self) -> None:
        path = shard_path(self._out_dir, self._split, self._next_index)
        self._current_file = path.open("wb", buffering=_WRITE_BUFFER_BYTES)
        self._next_index += 1
        self._tokens_in_current = 0

    def _close_current(self) -> None:
        if self._current_file is not None:
            self._current_file.close()
            self._current_file = None

    def __enter__(self) -> ShardWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
