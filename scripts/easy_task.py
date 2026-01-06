from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow `python scripts/easy_task.py ...` to import `scripts.run_experiment`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_experiment import main as run_experiment_main  # noqa: E402
from scripts.task_outputs import copy, ensure_clean_dir, remove_dir  # noqa: E402


def main() -> None:
    """Easy task: Lyrics (English vs Bangla) -> VAE -> KMeans -> t-SNE + metrics + PCA baseline."""

    p = argparse.ArgumentParser(description="Easy task runner (lyrics VAE clustering)")
    p.add_argument("--device", default="auto", help="auto|cpu|cuda")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--max_samples", type=int, default=2000, help="Per-language samples")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--tfidf_features", type=int, default=5000)

    args, unknown = p.parse_known_args()

    out_root = ensure_clean_dir(PROJECT_ROOT / "results" / "easy")
    tmp = ensure_clean_dir(out_root / "_tmp")

    sys.argv = [
        sys.argv[0],
        "--out_dir",
        str(tmp),
        "--modality",
        "lyrics",
        "--model",
        "vae",
        "--clusterer",
        "kmeans",
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--max_samples",
        str(args.max_samples),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--latent_dim",
        str(args.latent_dim),
        "--tfidf_features",
        str(args.tfidf_features),
    ]
    if args.no_amp:
        sys.argv.append("--no_amp")
    sys.argv.extend(unknown)

    run_experiment_main()

    # Rename/copy artifacts to match the desired folder layout
    copy(tmp / "latent_tsne_pred.png", out_root / "vae_tsne.png")
    copy(tmp / "baseline_tsne_pred.png", out_root / "pca_tsne.png")

    df = pd.read_csv(tmp / "clustering_metrics.csv")
    df.to_csv(out_root / "metrics_easy.csv", index=False)

    # Clean temp
    remove_dir(tmp)


if __name__ == "__main__":
    main()
