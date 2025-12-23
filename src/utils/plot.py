from __future__ import annotations

from pathlib import Path

import matplotlib


def set_matplotlib_backend() -> None:
    # Headless-safe backend.
    matplotlib.use("Agg", force=True)


def savefig(path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
