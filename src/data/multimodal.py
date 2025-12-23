from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import encode_labels, list_audio_items
from .lyrics import load_lyrics_dataset


@dataclass(frozen=True)
class MultiModalBatch:
    audio_paths: list[Path]
    texts: list[str]
    y_language: np.ndarray
    y_genre: np.ndarray


def make_synthetic_multimodal_dataset(
    audio_dir: Path,
    english_csv: Path,
    bangla_csv: Path,
    n_samples: int = 2000,
    seed: int = 42,
) -> MultiModalBatch:
    """Creates a simple paired audio+lyrics dataset.

    Your repo contains audio (GTZAN genres) and separate lyrics CSVs (English/Bangla)
    with no direct track-level alignment. For multimodal experiments, we create a
    *synthetic pairing* by randomly matching an audio track with a lyric sample.

    This still lets you test multimodal fusion, Beta-VAE representations, and
    evaluate clustering against known weak labels: genre (from audio filename)
    and language (from lyrics source).
    """

    rng = np.random.default_rng(seed)

    texts, y_lang = load_lyrics_dataset(english_csv, bangla_csv, max_samples_per_language=n_samples, seed=seed)

    audio_items = list_audio_items(audio_dir)
    if len(audio_items) == 0:
        raise FileNotFoundError(f"No .au files found in: {audio_dir}")

    audio_idx = rng.integers(0, len(audio_items), size=len(texts))
    chosen_audio = [audio_items[i] for i in audio_idx]

    y_genre, _ = encode_labels([a.genre for a in chosen_audio])

    return MultiModalBatch(
        audio_paths=[a.path for a in chosen_audio],
        texts=texts,
        y_language=y_lang,
        y_genre=y_genre,
    )
