from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    project_root: Path

    @property
    def dataset_dir(self) -> Path:
        return self.project_root / "Dataset"

    @property
    def audio_dir(self) -> Path:
        return self.dataset_dir / "Audio"

    @property
    def lyrics_en_csv(self) -> Path:
        return self.dataset_dir / "english.csv"

    @property
    def lyrics_bn_csv(self) -> Path:
        return self.dataset_dir / "Bangla.csv"

    @property
    def cache_dir(self) -> Path:
        return self.dataset_dir / "cache"

    @property
    def results_dir(self) -> Path:
        return self.project_root / "results"


def get_paths() -> Paths:
    # Assumes `src/` sits under project root.
    return Paths(project_root=Path(__file__).resolve().parents[1])
