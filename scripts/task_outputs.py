from __future__ import annotations

import shutil
from pathlib import Path


def ensure_clean_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def remove_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")
