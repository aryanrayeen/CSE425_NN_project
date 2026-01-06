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
    """Medium task: Audio features -> VAE -> clustering (kmeans/agglo/dbscan) + metrics + PCA baseline.

    Produces a single canonical folder: results/medium/
    """

    p = argparse.ArgumentParser(description="Medium task runner (audio VAE clustering)")
    p.add_argument("--device", default="auto", help="auto|cpu|cuda")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--audio_kind", choices=["mfcc", "mel"], default="mfcc")
    p.add_argument("--max_samples", type=int, default=1000)

    # For the guideline medium task, we keep this as a standard VAE.
    p.add_argument("--latent_dim", type=int, default=32)

    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=128)

    # DBSCAN
    p.add_argument("--eps", type=float, default=0.8)
    p.add_argument("--min_samples", type=int, default=5)

    args, unknown = p.parse_known_args()

    out_root = ensure_clean_dir(PROJECT_ROOT / "results" / "medium")

    metrics_frames: list[pd.DataFrame] = []

    def run_once(clusterer: str, extra: list[str], out_png_name: str) -> None:
        tmp = ensure_clean_dir(out_root / f"_tmp_{clusterer}")
        sys.argv = [
            sys.argv[0],
            "--out_dir",
            str(tmp),
            "--modality",
            "audio",
            "--audio_kind",
            args.audio_kind,
            "--model",
            "vae",
            "--clusterer",
            clusterer,
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
            *extra,
        ]
        if args.no_amp:
            sys.argv.append("--no_amp")
        sys.argv.extend(unknown)

        run_experiment_main()

        copy(tmp / "latent_tsne_pred.png", out_root / out_png_name)
        df = pd.read_csv(tmp / "clustering_metrics.csv")
        df.insert(0, "run", clusterer)
        metrics_frames.append(df)

        remove_dir(tmp)

    run_once("kmeans", [], "vae_kmeans_tsne.png")
    run_once("agglo", ["--linkage", "ward"], "vae_agglomerative_tsne.png")
    run_once("dbscan", ["--eps", str(args.eps), "--min_samples", str(args.min_samples)], "vae_dbscan_tsne.png")

    metrics = pd.concat(metrics_frames, ignore_index=True)
    metrics.to_csv(out_root / "metrics_medium.csv", index=False)


if __name__ == "__main__":
    main()
