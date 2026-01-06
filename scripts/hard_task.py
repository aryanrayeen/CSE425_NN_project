from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_experiment import main as run_experiment_main  # noqa: E402
from scripts.task_outputs import copy, ensure_clean_dir, remove_dir  # noqa: E402


def main() -> None:
    """Hard task: Multimodal fusion (audio MFCC + lyrics TF-IDF) -> Beta-VAE -> clustering + ARI/NMI/Purity."""

    p = argparse.ArgumentParser(description="Hard task runner (multimodal Beta-VAE clustering)")
    p.add_argument("--device", default="auto", help="auto|cpu|cuda")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--clusterer", choices=["kmeans", "agglo", "dbscan"], default="kmeans")
    p.add_argument("--eval_label", choices=["language", "genre", "both"], default="both")

    p.add_argument("--max_samples", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)

    p.add_argument("--latent_dim", type=int, default=32)
    p.add_argument("--beta", type=float, default=6.0)
    p.add_argument("--tfidf_features", type=int, default=5000)

    # DBSCAN
    p.add_argument("--eps", type=float, default=0.8)
    p.add_argument("--min_samples", type=int, default=5)

    args, unknown = p.parse_known_args()

    out_root = ensure_clean_dir(PROJECT_ROOT / "results" / "hard")
    tmp = ensure_clean_dir(out_root / "_tmp")

    sys.argv = [
        sys.argv[0],
        "--out_dir",
        str(tmp),
        "--modality",
        "multimodal",
        "--model",
        "beta_vae",
        "--clusterer",
        args.clusterer,
        "--eval_label",
        args.eval_label,
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
        "--beta",
        str(args.beta),
        "--tfidf_features",
        str(args.tfidf_features),
        "--eps",
        str(args.eps),
        "--min_samples",
        str(args.min_samples),
    ]
    if args.no_amp:
        sys.argv.append("--no_amp")
    sys.argv.extend(unknown)

    run_experiment_main()

    copy(tmp / "latent_tsne_pred.png", out_root / "beta_vae_tsne.png")
    df = pd.read_csv(tmp / "clustering_metrics.csv")
    df.to_csv(out_root / "metrics_hard.csv", index=False)

    remove_dir(tmp)


if __name__ == "__main__":
    main()
