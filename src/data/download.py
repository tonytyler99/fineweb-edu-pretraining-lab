"""
 download wrapper for the FineWeb-Edu dataset.

"""

from __future__ import annotations

from collections.abc import Iterator

from datasets import load_dataset

FINEWEB_EDU_REPO = "HuggingFaceFW/fineweb-edu"


def iter_fineweb_edu_docs(
    subset: str = "sample-10BT",
    split: str = "train",
    max_docs: int | None = None,
) -> Iterator[str]:
    """Yield FineWeb-Edu document texts from a streaming HF dataset.

    Args:
        subset: HF dataset config name. Defaults to ``"sample-10BT"`` (~10 B
            GPT-2 tokens),
        split: Only ``"train"`` is published for FineWeb-Edu.
        max_docs: Cap on document count for dry-runs / tests. ``None`` (default)

    Yields:
        Raw UTF-8 document texts. No tokenization, normalization, or EOT
        injection is done here; tokenization consumers are responsible for
        adding document separators.
    """
    stream = load_dataset(FINEWEB_EDU_REPO, name=subset, split=split, streaming=True)

    for i, sample in enumerate(stream):
        if max_docs is not None and i >= max_docs:
            return
        yield sample["text"]
