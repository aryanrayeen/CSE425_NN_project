from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_SAVED_RE = re.compile(r"^Saved results to:\s*(.*)\s*$")


@dataclass(frozen=True)
class Experiment:
    name: str
    args: list[str]


def _run_one(py: str, cwd: Path, exp: Experiment) -> Path:
    cmd = [py, "scripts/run_experiment.py", *exp.args]
    print("\n===", exp.name, "===")
    print(" ".join(cmd))

    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
    )

    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"Experiment failed: {exp.name} (exit={proc.returncode})")

    out_dir: Path | None = None
    for line in proc.stdout.splitlines():
        m = _SAVED_RE.match(line.strip())
        if m:
            out_dir = Path(m.group(1).strip())
            break

    # Echo logs for visibility
    print(proc.stdout)
    if proc.stderr.strip():
        print(proc.stderr, file=sys.stderr)

    if out_dir is None:
        raise SystemExit(f"Could not parse results directory from output for: {exp.name}")

    if not out_dir.exists():
        raise SystemExit(f"Parsed results directory does not exist: {out_dir}")

    return out_dir


def _read_metrics_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(dict(r))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    cols: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in cols:
                cols.append(k)

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_experiments(args: argparse.Namespace) -> list[Experiment]:
    common = [
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]
    if args.no_amp:
        common.append("--no_amp")

    # Easy: lyrics (English vs Bangla)
    easy = [
        Experiment(
            name="easy_lyrics_vae_kmeans",
            args=[
                "--modality",
                "lyrics",
                "--model",
                "vae",
                "--clusterer",
                "kmeans",
                "--max_samples",
                str(args.lyrics_samples),
                "--epochs",
                str(args.lyrics_epochs),
                "--batch_size",
                str(args.batch_size),
                *common,
            ],
        ),
    ]

    # Medium: audio MFCC with multiple clusterers
    medium = [
        Experiment(
            name="medium_audio_mfcc_vae_kmeans",
            args=[
                "--modality",
                "audio",
                "--audio_kind",
                "mfcc",
                "--model",
                "vae",
                "--clusterer",
                "kmeans",
                "--max_samples",
                str(args.audio_samples),
                "--epochs",
                str(args.audio_epochs),
                "--batch_size",
                str(args.batch_size),
                *common,
            ],
        ),
        Experiment(
            name="medium_audio_mfcc_vae_agglo",
            args=[
                "--modality",
                "audio",
                "--audio_kind",
                "mfcc",
                "--model",
                "vae",
                "--clusterer",
                "agglo",
                "--max_samples",
                str(args.audio_samples),
                "--epochs",
                str(args.audio_epochs),
                "--batch_size",
                str(args.batch_size),
                *common,
            ],
        ),
        Experiment(
            name="medium_audio_mfcc_vae_dbscan",
            args=[
                "--modality",
                "audio",
                "--audio_kind",
                "mfcc",
                "--model",
                "vae",
                "--clusterer",
                "dbscan",
                "--eps",
                str(args.dbscan_eps),
                "--min_samples",
                str(args.dbscan_min_samples),
                "--max_samples",
                str(args.audio_samples),
                "--epochs",
                str(args.audio_epochs),
                "--batch_size",
                str(args.batch_size),
                *common,
            ],
        ),
    ]

    # Medium+: Conv-VAE on mel spectrograms
    conv = [
        Experiment(
            name="medium_audio_mel_conv_vae_kmeans",
            args=[
                "--modality",
                "audio",
                "--audio_kind",
                "mel",
                "--model",
                "conv_vae",
                "--clusterer",
                "kmeans",
                "--latent_dim",
                str(args.conv_latent_dim),
                "--beta",
                str(args.beta),
                "--max_samples",
                str(args.conv_audio_samples),
                "--epochs",
                str(args.conv_epochs),
                "--batch_size",
                str(args.conv_batch_size),
                *common,
            ],
        )
    ]

    # Hard: multimodal fusion + Beta-VAE
    hard = [
        Experiment(
            name="hard_multimodal_beta_vae_kmeans",
            args=[
                "--modality",
                "multimodal",
                "--model",
                "beta_vae",
                "--beta",
                str(args.beta),
                "--latent_dim",
                str(args.latent_dim),
                "--clusterer",
                "kmeans",
                "--eval_label",
                "both",
                "--max_samples",
                str(args.multimodal_samples),
                "--epochs",
                str(args.multimodal_epochs),
                "--batch_size",
                str(args.batch_size),
                *common,
            ],
        )
    ]

    exps = []
    if args.which in ("all", "easy"):
        exps += easy
    if args.which in ("all", "medium"):
        exps += medium
        exps += conv
    if args.which in ("all", "hard"):
        exps += hard

    return exps


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the full suite of experiments and aggregate metrics.")
    ap.add_argument("--which", choices=["all", "easy", "medium", "hard"], default="all")

    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--batch_size", type=int, default=128)

    ap.add_argument("--lyrics_samples", type=int, default=2000, help="Per-language")
    ap.add_argument("--lyrics_epochs", type=int, default=10)

    ap.add_argument("--audio_samples", type=int, default=1000)
    ap.add_argument("--audio_epochs", type=int, default=15)

    ap.add_argument("--dbscan_eps", type=float, default=0.8)
    ap.add_argument("--dbscan_min_samples", type=int, default=5)

    ap.add_argument("--conv_audio_samples", type=int, default=400)
    ap.add_argument("--conv_epochs", type=int, default=20)
    ap.add_argument("--conv_batch_size", type=int, default=64)
    ap.add_argument("--conv_latent_dim", type=int, default=32)

    ap.add_argument("--multimodal_samples", type=int, default=2000)
    ap.add_argument("--multimodal_epochs", type=int, default=20)

    ap.add_argument("--latent_dim", type=int, default=32)
    ap.add_argument("--beta", type=float, default=6.0)

    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    py = sys.executable

    out_dirs: list[Path] = []
    metrics_rows: list[dict[str, Any]] = []

    for exp in build_experiments(args):
        out_dir = _run_one(py, project_root, exp)
        out_dirs.append(out_dir)
        metrics_path = out_dir / "clustering_metrics.csv"
        if metrics_path.exists():
            rows = _read_metrics_csv(metrics_path)
            for r in rows:
                r["run_name"] = exp.name
                r["run_dir"] = str(out_dir)
            metrics_rows.extend(rows)

    summary_dir = project_root / "results" / "_summary"
    summary_path = summary_dir / "all_runs_metrics.csv"
    _write_csv(summary_path, metrics_rows)

    print("\nAll experiments finished.")
    print("Aggregated metrics:", summary_path)
    print("Run folders:")
    for d in out_dirs:
        print(" -", d)


if __name__ == "__main__":
    main()
