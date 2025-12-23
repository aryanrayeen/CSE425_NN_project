from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.text import normalize_text


@dataclass(frozen=True)
class LyricsSample:
    text: str
    language: str  # "en" or "bn"


def _load_english(csv_path: Path, max_samples: int | None, min_chars: int) -> list[LyricsSample]:
    df = pd.read_csv(csv_path)
    # Expected: Name,Artist,Album,Popularity,Lyrics
    col = "Lyrics" if "Lyrics" in df.columns else df.columns[-1]

    texts: list[str] = []
    for t in df[col].astype(str).tolist():
        t = normalize_text(t)
        if len(t) >= min_chars:
            texts.append(t)

    if max_samples is not None:
        texts = texts[: max_samples]

    return [LyricsSample(text=t, language="en") for t in texts]


def _load_bangla(csv_path: Path, max_samples: int | None, min_chars: int) -> list[LyricsSample]:
    df = pd.read_csv(csv_path)
    # Expected: song_name,genre,lyrics,artist_name
    col = "lyrics" if "lyrics" in df.columns else df.columns[-1]

    texts: list[str] = []
    for t in df[col].astype(str).tolist():
        t = normalize_text(t)
        if len(t) >= min_chars:
            texts.append(t)

    if max_samples is not None:
        texts = texts[: max_samples]

    return [LyricsSample(text=t, language="bn") for t in texts]


def load_lyrics_dataset(
    english_csv: Path,
    bangla_csv: Path,
    max_samples_per_language: int = 2000,
    min_chars: int = 50,
    seed: int = 42,
) -> tuple[list[str], np.ndarray]:
    """Returns texts + language labels (0=en, 1=bn)."""

    en = _load_english(english_csv, max_samples_per_language, min_chars)
    bn = _load_bangla(bangla_csv, max_samples_per_language, min_chars)

    samples = en + bn
    rng = np.random.default_rng(seed)
    rng.shuffle(samples)

    texts = [s.text for s in samples]
    y = np.array([0 if s.language == "en" else 1 for s in samples], dtype=np.int64)
    return texts, y
