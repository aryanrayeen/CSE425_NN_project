from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/run_experiment.py ...` to import `src.*`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.baselines import pca_kmeans
from src.clustering import cluster_agglo, cluster_dbscan, cluster_kmeans
from src.config import get_paths
from src.data.audio import encode_labels, list_audio_items
from src.data.lyrics import load_lyrics_dataset
from src.data.multimodal import make_synthetic_multimodal_dataset
from src.evaluation import evaluate_clustering
from src.features.audio_features import AudioFeatureConfig, cached_audio_features
from src.features.fusion import concat_modalities, zscore
from src.features.lyrics_features import fit_tfidf
from src.reconstruction import save_conv_recon_examples
from src.training import (
    TrainConfig,
    encode_latents_conv_vae,
    encode_latents_mlp_ae,
    encode_latents_mlp_vae,
    train_conv_vae,
    train_mlp_ae,
    train_mlp_vae,
)
from src.utils.device import pick_device
from src.utils.io import ensure_dir, save_json
from src.utils.seed import seed_everything
from src.utils.time import now_tag
from src.visualize import embed_2d, plot_embedding


def _pick_clusterer(args, x_embed: np.ndarray, n_clusters: int) -> np.ndarray:
    if args.clusterer == "kmeans":
        return cluster_kmeans(x_embed, n_clusters=n_clusters, seed=args.seed).labels
    if args.clusterer == "agglo":
        return cluster_agglo(x_embed, n_clusters=n_clusters, linkage=args.linkage).labels
    if args.clusterer == "dbscan":
        return cluster_dbscan(x_embed, eps=args.eps, min_samples=args.min_samples).labels
    raise ValueError(f"Unknown clusterer: {args.clusterer}")


def _infer_n_clusters(modality: str, y_true: np.ndarray | None, y_true_alt: np.ndarray | None) -> int:
    if modality == "lyrics":
        return 2
    if modality == "audio":
        return int(len(np.unique(y_true))) if y_true is not None else 10
    if modality == "multimodal":
        if y_true is not None:
            return int(len(np.unique(y_true)))
        if y_true_alt is not None:
            return int(len(np.unique(y_true_alt)))
        return 20
    return 10


def main() -> None:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError as e:
        raise SystemExit(
            "PyTorch (torch) is not installed for the Python you are using.\n"
            "Fix (recommended):\n"
            "  python -m pip install -r requirements.txt\n"
            "If you are using the repo venv explicitly:\n"
            "  C:/Users/User/Documents/CSE425_NN_project/.venv/Scripts/python.exe -m pip install -r requirements.txt\n"
            "Then re-run:\n"
            "  python scripts/run_experiment.py --modality lyrics --model vae --clusterer kmeans --device auto\n"
        ) from e

    # Import torch-dependent modules only after torch is confirmed installed.
    from src.models import ConvVAE, MLPAutoEncoder, MLPVAE

    p = argparse.ArgumentParser(description="VAE/Beta-VAE clustering for audio+lyrics datasets")
    p.add_argument(
        "--out_dir",
        default="",
        help="If set, write outputs to this directory instead of creating a timestamped folder under results/.",
    )
    p.add_argument("--modality", choices=["lyrics", "audio", "multimodal"], default="lyrics")
    p.add_argument("--model", choices=["vae", "beta_vae", "ae", "conv_vae"], default="vae")
    p.add_argument("--audio_kind", choices=["mfcc", "mel"], default="mfcc")
    p.add_argument("--clusterer", choices=["kmeans", "agglo", "dbscan"], default="kmeans")
    p.add_argument("--n_clusters", type=int, default=0, help="0 = infer")

    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--beta", type=float, default=4.0, help="Used for beta_vae/conv_vae")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--max_samples", type=int, default=2000, help="Per-language for lyrics; total for audio")
    p.add_argument("--tfidf_features", type=int, default=5000)

    p.add_argument("--device", default="auto", help="auto|cpu|cuda")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    # Agglo
    p.add_argument("--linkage", default="ward", choices=["ward", "complete", "average", "single"])
    # DBSCAN
    p.add_argument("--eps", type=float, default=0.8)
    p.add_argument("--min_samples", type=int, default=5)

    # Multimodal evaluation label
    p.add_argument("--eval_label", choices=["language", "genre", "both"], default="both")

    args = p.parse_args()

    seed_everything(args.seed)
    dev = pick_device(args.device, amp=not args.no_amp)

    paths = get_paths()
    ensure_dir(paths.cache_dir)
    ensure_dir(paths.results_dir)

    if args.out_dir:
        out_dir = ensure_dir(Path(args.out_dir))
    else:
        tag = now_tag()
        out_dir = ensure_dir(paths.results_dir / f"{tag}_{args.modality}_{args.model}_{args.clusterer}")

    # -------------------- Load data + features --------------------
    y_true: np.ndarray | None = None
    y_true_alt: np.ndarray | None = None

    x_raw: np.ndarray
    x_for_baseline: np.ndarray

    if args.modality == "lyrics":
        texts, y_true = load_lyrics_dataset(paths.lyrics_en_csv, paths.lyrics_bn_csv, max_samples_per_language=args.max_samples, seed=args.seed)
        x_raw, _vec = fit_tfidf(texts, max_features=args.tfidf_features)
        x_raw, _, _ = zscore(x_raw)
        x_for_baseline = x_raw

    elif args.modality == "audio":
        items = list_audio_items(paths.audio_dir, limit=args.max_samples)
        if len(items) == 0:
            raise FileNotFoundError(f"No audio files found at {paths.audio_dir}")
        audio_paths = [it.path for it in items]
        y_true, _mapping = encode_labels([it.genre for it in items])

        cfg = AudioFeatureConfig()
        if args.model == "conv_vae" or args.audio_kind == "mel":
            x_raw = cached_audio_features(audio_paths, paths.cache_dir, kind="mel", cfg=cfg)
            x_for_baseline = x_raw.reshape(x_raw.shape[0], -1)
        else:
            x_raw = cached_audio_features(audio_paths, paths.cache_dir, kind="mfcc", cfg=cfg)
            x_raw, _, _ = zscore(x_raw)
            x_for_baseline = x_raw

    elif args.modality == "multimodal":
        batch = make_synthetic_multimodal_dataset(paths.audio_dir, paths.lyrics_en_csv, paths.lyrics_bn_csv, n_samples=args.max_samples, seed=args.seed)

        x_text, _vec = fit_tfidf(batch.texts, max_features=args.tfidf_features)
        x_text, _, _ = zscore(x_text)

        cfg = AudioFeatureConfig()
        x_audio = cached_audio_features(batch.audio_paths, paths.cache_dir, kind="mfcc", cfg=cfg)
        x_audio, _, _ = zscore(x_audio)

        x_raw = concat_modalities(x_audio, x_text)
        x_raw, _, _ = zscore(x_raw)
        x_for_baseline = x_raw

        if args.eval_label == "language":
            y_true = batch.y_language
        elif args.eval_label == "genre":
            y_true = batch.y_genre
        else:
            # Combine (genre, language) -> unique class id
            y_true = (batch.y_genre.astype(np.int64) * 2 + batch.y_language.astype(np.int64)).astype(np.int64)
            y_true_alt = batch.y_genre

    else:
        raise ValueError(f"Unknown modality: {args.modality}")

    # -------------------- Train representation model --------------------
    train_cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, beta=(args.beta if args.model in {"beta_vae", "conv_vae"} else 1.0))

    if args.model in {"vae", "beta_vae"}:
        if x_raw.ndim != 2:
            x_vec = x_raw.reshape(x_raw.shape[0], -1)
        else:
            x_vec = x_raw
        model = MLPVAE(input_dim=x_vec.shape[1], latent_dim=args.latent_dim)
        history = train_mlp_vae(x_vec, model, train_cfg, dev)
        z = encode_latents_mlp_vae(x_vec, model, dev)

    elif args.model == "ae":
        if x_raw.ndim != 2:
            x_vec = x_raw.reshape(x_raw.shape[0], -1)
        else:
            x_vec = x_raw
        model = MLPAutoEncoder(input_dim=x_vec.shape[1], latent_dim=args.latent_dim)
        history = train_mlp_ae(x_vec, model, train_cfg, dev)
        z = encode_latents_mlp_ae(x_vec, model, dev)

    elif args.model == "conv_vae":
        if x_raw.ndim != 4:
            raise ValueError("conv_vae requires mel features with shape (N,1,n_mels,n_frames). Use --audio_kind mel or modality=audio.")
        model = ConvVAE(n_mels=x_raw.shape[2], n_frames=x_raw.shape[3], latent_dim=args.latent_dim)
        history = train_conv_vae(x_raw, model, train_cfg, dev)
        z = encode_latents_conv_vae(x_raw, model, dev)

        # Save a small reconstruction grid
        import torch

        model.eval()
        xb = torch.from_numpy(x_raw[: min(16, x_raw.shape[0])]).to(dev.device)
        with torch.no_grad():
            recon, _mu, _logvar, _ = model(xb)
        save_conv_recon_examples(x_raw[: recon.shape[0]], recon.detach().cpu().numpy(), out_dir / "recon_examples.png")

    else:
        raise ValueError(f"Unknown model: {args.model}")

    # -------------------- Clustering + baselines --------------------
    n_clusters = args.n_clusters if args.n_clusters > 0 else _infer_n_clusters(args.modality, y_true, y_true_alt)

    y_pred = _pick_clusterer(args, z, n_clusters=n_clusters)
    rep_report = evaluate_clustering(z, y_pred, y_true=y_true)

    baseline = pca_kmeans(x_for_baseline, n_components=min(32, x_for_baseline.shape[1]), n_clusters=n_clusters, seed=args.seed)
    base_report = evaluate_clustering(baseline.embeddings, baseline.cluster_labels, y_true=y_true)

    rows: list[dict[str, object]] = []
    rows.append({"representation": "latent", **rep_report.metrics, "clusterer": args.clusterer, "model": args.model, "modality": args.modality, "device": dev.device, "amp": dev.amp})
    rows.append({"representation": baseline.name, **base_report.metrics, "clusterer": "kmeans", "model": "pca", "modality": args.modality, "device": "cpu", "amp": False})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "clustering_metrics.csv", index=False)

    # -------------------- Visualization --------------------
    emb2d = embed_2d(z, method="tsne", seed=args.seed)
    plot_embedding(emb2d, y_pred, title=f"{args.modality} | {args.model} | {args.clusterer}", out_path=out_dir / "latent_tsne_pred.png")
    if y_true is not None:
        plot_embedding(emb2d, y_true, title=f"{args.modality} | true labels", out_path=out_dir / "latent_tsne_true.png")

    base2d = embed_2d(baseline.embeddings, method="tsne", seed=args.seed)
    plot_embedding(base2d, baseline.cluster_labels, title=f"baseline PCA+KMeans", out_path=out_dir / "baseline_tsne_pred.png")

    # -------------------- Save run config --------------------
    save_json(
        out_dir / "run_config.json",
        {
            "args": vars(args),
            "device": {"device": dev.device, "amp": dev.amp},
            "train_history": history,
            "n_clusters": int(n_clusters),
        },
    )

    print("Saved results to:", out_dir)
    print(df)


if __name__ == "__main__":
    main()
