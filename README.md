# CSE425 Unsupervised Learning Project: VAE for Hybrid Language Music Clustering

This repo implements the project described in [Guidelines.md](Guidelines.md):
- **Easy**: basic VAE → latent features → **K-Means** clustering + **t-SNE** visualization + baseline (**PCA + K-Means**) and metrics.
- **Medium**: audio **MFCC** features + optional **Conv-VAE** on mel-spectrograms, multiple clusterers (K-Means/Agglo/DBSCAN), more metrics.
- **Hard**: **Beta-VAE** (disentangled latent) and **multimodal fusion** (audio + lyrics) with label-based metrics (ARI/NMI/Purity) when labels are available.

Your datasets live in `Dataset/`:
- `Dataset/english.csv` (English lyrics)
- `Dataset/Bangla.csv` (Bangla lyrics)
- `Dataset/Audio/*.au` (GTZAN-style audio by genre: `blues.00000.au`, ...)

## Setup (Windows)

### 1) Install dependencies

Use `python -m pip` (works even if `pip` is not on PATH):

```powershell
python -m pip install -r requirements.txt
```

If you want to be explicit about the repo venv:

```powershell
C:/Users/User/Documents/CSE425_NN_project/.venv/Scripts/python.exe -m pip install -r requirements.txt
```

### 2) Install PyTorch with CUDA (GPU)

Pick the command for your CUDA version from https://pytorch.org/get-started/locally/.
Example (CUDA 12.1):

```powershell
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch
```

If you don’t have an NVIDIA GPU, PyTorch will fall back to CPU automatically.

## Run Experiments

All runs save outputs under `results/` (metrics CSV + plots + config JSON).

### Task entrypoints (easy / medium / hard)

These scripts run the guideline tasks directly:

- Easy: `python scripts/easy_task.py --device auto`
- Medium: `python scripts/medium_task.py --device auto`
- Hard: `python scripts/hard_task.py --device auto`

They overwrite a single canonical output folder each:
- `results/easy/` (e.g., `metrics_easy.csv`, `vae_tsne.png`, `pca_tsne.png`)
- `results/medium/` (e.g., `metrics_medium.csv`, `vae_kmeans_tsne.png`, `vae_agglomerative_tsne.png`, `vae_dbscan_tsne.png`)
- `results/hard/` (e.g., `metrics_hard.csv`, `beta_vae_tsne.png`)

### Easy task (lyrics: English vs Bangla)

VAE latent → KMeans(2) + baseline PCA+KMeans:

```powershell
python scripts/run_experiment.py --modality lyrics --model vae --clusterer kmeans --max_samples 2000 --epochs 10
```

### Medium task (audio MFCC)

```powershell
python scripts/run_experiment.py --modality audio --audio_kind mfcc --model vae --clusterer kmeans --max_samples 1000 --epochs 15
```

Try other clusterers:

```powershell
python scripts/run_experiment.py --modality audio --audio_kind mfcc --model vae --clusterer agglo --max_samples 1000
python scripts/run_experiment.py --modality audio --audio_kind mfcc --model vae --clusterer dbscan --eps 0.8 --min_samples 5 --max_samples 1000
```

### Medium+ (Conv-VAE on mel-spectrograms)

```powershell
python scripts/run_experiment.py --modality audio --audio_kind mel --model conv_vae --latent_dim 32 --beta 4.0 --epochs 20 --batch_size 64
```

This also saves a reconstruction grid: `recon_examples.png`.

### Hard task (Beta-VAE + multimodal fusion)

Your audio and lyrics files don’t have track-level alignment. For multimodal experiments, we create a **synthetic pairing** by randomly matching an audio clip with a lyric sample. This still supports multimodal fusion experiments and label-based evaluations (genre/language).

```powershell
python scripts/run_experiment.py --modality multimodal --model beta_vae --beta 6.0 --latent_dim 32 --clusterer kmeans --eval_label both --max_samples 2000 --epochs 20
```

## Outputs

Each run creates a folder under `results/` containing:
- `clustering_metrics.csv` (Silhouette, Calinski-Harabasz, Davies-Bouldin, and ARI/NMI/Purity when labels are provided)
- `latent_tsne_pred.png` (t-SNE colored by predicted clusters)
- `latent_tsne_true.png` (if labels exist)
- `baseline_tsne_pred.png` (PCA+KMeans baseline)
- `run_config.json` (all arguments + train history)

## Notes on GPU

The training code uses:
- `--device auto` (default) → uses CUDA if available
- AMP mixed precision on GPU unless you pass `--no_amp`

You can force CPU:

```powershell
python scripts/run_experiment.py --device cpu ...
```
