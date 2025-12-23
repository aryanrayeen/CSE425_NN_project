from __future__ import annotations

import re


_whitespace = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text.replace("\u200b", " ")
    text = text.replace("\r", " ").replace("\n", " ")
    text = _whitespace.sub(" ", text).strip()
    return text
