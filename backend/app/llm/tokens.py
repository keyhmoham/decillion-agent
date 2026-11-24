"""Token counting for context management.

Uses the model's real tokenizer (``tokenizer.json`` of Agents-A1-4B, fetched at image build time
by ``scripts/fetch_tokenizer.py``) and falls back to a conservative heuristic when absent.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

log = logging.getLogger(__name__)


class TokenCounter:
    def __init__(self, tokenizer_path: str | None = None) -> None:
        self._tok = None
        if tokenizer_path and os.path.exists(tokenizer_path):
            try:
                from tokenizers import Tokenizer

                self._tok = Tokenizer.from_file(tokenizer_path)
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not load tokenizer %s: %s", tokenizer_path, exc)

    @property
    def exact(self) -> bool:
        return self._tok is not None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._tok is not None:
            return len(self._tok.encode(text, add_special_tokens=False).ids)
        # ~3.3 chars/token for English prose+code on this vocab; err on the high side.
        return int(len(text) / 3.2) + 1


@lru_cache(maxsize=4)
def get_counter(path: str | None = None) -> TokenCounter:
    if path is None:
        from ..config import get_settings

        path = get_settings().tokenizer_path
    return TokenCounter(path)
