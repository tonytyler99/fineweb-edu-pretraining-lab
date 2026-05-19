"""GPT-2 BPE tokenization helpers for pretraining shards.

This file was renamed from `tokenize.py` to avoid shadowing the standard
library ``tokenize`` module which caused circular import errors at runtime.
"""

from __future__ import annotations
from collections.abc import Iterable, Iterator
from typing import Final


import numpy as np
import tiktoken

GPT2_ENCODING_NAME : Final = "gpt2" # GPT-2 BPE was selected for model performance
GPT2_VOCAB_SIZE =  50_257


# Module-level singleton; tiktoken caches BPE merges to disk on first call.
_encoder = tiktoken.get_encoding(GPT2_ENCODING_NAME)


def encode_doc(text: str) -> np.ndarray:
    """Encode a document and append the EOT separator.

    Args:
        text: Raw UTF-8 document text.

    Returns:
        1-D ``uint16`` array
    """
    # encode_ordinary skips special-token bookkeeping; we control EOT ourselves
    # so a literal ``<|endoftext|>`` in raw text cannot inject a separator.
    tokens = _encoder.encode_ordinary(text)
    tokens.append(_encoder.eot_token)
    return np.asarray(tokens, dtype=np.uint16)


def iter_encoded(docs: Iterable[str]) -> Iterator[np.ndarray]:
    """Stream-encode an iterable of documents, one array per doc.

    Args:
        docs: Iterable of raw document texts.

    Yields:
        ``uint16`` arrays.
    """
    for doc in docs:
        yield encode_doc(doc)
