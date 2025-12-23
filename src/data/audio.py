from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AudioItem:
    path: Path
    genre: str


def list_audio_items(audio_dir: Path, limit: int | None = None) -> list[AudioItem]:
    paths = sorted(p for p in audio_dir.glob("*.au") if p.is_file())
    if limit is not None:
        paths = paths[:limit]

    items: list[AudioItem] = []
    for p in paths:
        # GTZAN-style naming: genre.00000.au
        genre = p.name.split(".")[0]
        items.append(AudioItem(path=p, genre=genre))
    return items


def encode_labels(labels: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    uniq = sorted(set(labels))
    mapping = {k: i for i, k in enumerate(uniq)}
    y = np.array([mapping[x] for x in labels], dtype=np.int64)
    return y, mapping
