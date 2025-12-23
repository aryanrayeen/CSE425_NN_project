from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..utils.io import ensure_dir, load_npz, save_npz


@dataclass(frozen=True)
class AudioFeatureConfig:
    sample_rate: int = 22050
    clip_seconds: float = 30.0
    n_mfcc: int = 20
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    fixed_frames: int = 256


def _load_audio_mono(path: Path, sample_rate: int, clip_seconds: float) -> np.ndarray:
    import librosa

    y, sr = librosa.load(path.as_posix(), sr=sample_rate, mono=True)
    target_len = int(sample_rate * clip_seconds)
    if y.shape[0] < target_len:
        y = np.pad(y, (0, target_len - y.shape[0]))
    else:
        y = y[:target_len]
    return y.astype(np.float32)


def extract_mfcc_vector(path: Path, cfg: AudioFeatureConfig) -> np.ndarray:
    import librosa

    y = _load_audio_mono(path, cfg.sample_rate, cfg.clip_seconds)
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=cfg.sample_rate,
        n_mfcc=cfg.n_mfcc,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    )
    # Aggregate over time: mean + std
    mu = mfcc.mean(axis=1)
    sd = mfcc.std(axis=1)
    feat = np.concatenate([mu, sd], axis=0)
    return feat.astype(np.float32)


def extract_mel_spectrogram(path: Path, cfg: AudioFeatureConfig) -> np.ndarray:
    import librosa

    y = _load_audio_mono(path, cfg.sample_rate, cfg.clip_seconds)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=cfg.sample_rate,
        n_mels=cfg.n_mels,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    # Normalize roughly to [0, 1]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

    # Fix time dimension
    frames = mel_db.shape[1]
    if frames < cfg.fixed_frames:
        mel_db = np.pad(mel_db, ((0, 0), (0, cfg.fixed_frames - frames)))
    else:
        mel_db = mel_db[:, : cfg.fixed_frames]

    # shape: (1, n_mels, fixed_frames)
    return mel_db[None, ...].astype(np.float32)


def cached_audio_features(
    paths: list[Path],
    cache_dir: Path,
    kind: str,
    cfg: AudioFeatureConfig,
) -> np.ndarray:
    ensure_dir(cache_dir)
    cache_path = cache_dir / f"audio_{kind}_sr{cfg.sample_rate}_clip{int(cfg.clip_seconds)}_mfcc{cfg.n_mfcc}_mels{cfg.n_mels}_frames{cfg.fixed_frames}.npz"

    key = "x"
    key_paths = "paths"

    if cache_path.exists():
        try:
            data = load_npz(cache_path)
            cached = data.get(key)
            cached_paths = data.get(key_paths)
            if cached is not None and cached_paths is not None:
                cached_paths_list = [p for p in cached_paths.astype(str).tolist()]
                if cached_paths_list == [p.as_posix() for p in paths]:
                    return cached
        except ValueError:
            # Older cache versions may store `paths` as object arrays, which can't be
            # loaded with allow_pickle=False. Recompute and overwrite.
            pass

    feats: list[np.ndarray] = []
    for p in paths:
        if kind == "mfcc":
            feats.append(extract_mfcc_vector(p, cfg))
        elif kind == "mel":
            feats.append(extract_mel_spectrogram(p, cfg))
        else:
            raise ValueError(f"Unknown kind: {kind}")

    x = np.stack(feats, axis=0)
    save_npz(cache_path, **{key: x, key_paths: np.array([p.as_posix() for p in paths], dtype=np.str_)})
    return x
