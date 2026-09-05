# SHARAD Radar Denoising

Deep learning denoising of Mars SHARAD orbital radar using a residual U-Net, with an ongoing extension toward self-supervised (Noise2Self-style) denoising for real radargrams where no clean reference exists.

## Motivation

SHARAD (Shallow Radar) subsurface echoes are contaminated by systematic noise picked up from the bottom of each orbital track. There is no clean ground-truth radargram for real Mars data, so this project first builds a fully supervised pipeline on synthetically noised pairs, and uses it as the foundation for a self-supervised extension that can be evaluated directly on real, unlabeled tracks.

## Pipeline overview

1. **Preprocess** raw PDS3 RGRAM products into aligned reflection/noise pairs (`data_process.py`)
2. **QC** the resulting dataset and derive a normalization scheme (`check_npz_stats.py`)
3. **Split** by orbital track to prevent leakage (`split_dataset.py`)
4. **Load & augment** patches on the fly for training (`dataset.py`)
5. **Train** a residual U-Net to predict noise for subtraction (`model_unet.py`, `train_unet.py`)
6. **Run full-image inference** with sliding-window overlap-add (`infer_full_image.py`)
7. **Project** denoised patches back onto the original RGRAM image (`project_denoised_to_rgram.py`)
8. **Geolocate** tracks for context and figure-making (`polar_picture_draw.py`, `radar_tracks_mapping.py`, `top_correct.py`)

## Repository structure

| File | Purpose |
|---|---|
| `data_process.py` | Parses SHARAD RGRAM `.LBL`/`.IMG` (PDS3), picks the surface per column, subtracts bottom-track noise mean, extracts a surface-aligned reflection window and a matched noise patch from the track's own bottom region, and writes per-track training samples to compressed `.npz` |
| `check_npz_stats.py` | Validates every `.npz` (required keys, shapes), computes per-file and global statistics, and recommends log1p + percentile (p1/p99) normalization parameters |
| `split_dataset.py` | Splits the dataset into train/val/test **by orbital track**, then verifies no track appears in more than one split |
| `dataset.py` | PyTorch `Dataset`/`DataLoader`: random patch cropping, dynamic re-noising (fresh noise scale drawn per sample) so each clean patch is exposed to many noise realizations, log1p + percentile normalization |
| `model_unet.py` | `ResidualUNet` — predicts the noise component; the denoised output is computed as `input - predicted_noise`. Configurable depth, base channels, GroupNorm, bilinear upsampling |
| `train_unet.py` | Trains with a masked SmoothL1 loss between the denoised output and the clean reflection target; tracks MAE/MSE/PSNR each epoch; AdamW + AMP + gradient clipping; saves checkpoints, periodic previews, and a CSV log |
| `infer_full_image.py` | Sliding-window inference over full-width radargrams with weighted overlap-add blending, then denormalizes back to the original intensity scale |
| `project_denoised_to_rgram.py` | Maps the denoised, surface-aligned window back onto the original full RGRAM image using the stored per-column surface picks |
| `polar_picture_draw.py`, `radar_tracks_mapping.py` | Project SHARAD ground-track latitude/longitude onto a Mars north-polar stereographic grid over a MOLA DEM, for spatial context and track selection |
| `top_correct.py` | Standalone column-shift / topographic correction utility for producing corrected radargram figures, independent of the main U-Net pipeline |
| `test_single_rgram_process.ipynb` | Interactive single-track walkthrough of the preprocessing steps, used for debugging and visual inspection before batch processing |
| `environment.yml` | Conda environment specification |

## Method details

**Preprocessing.** For each track, the surface is picked per column with a sliding-window amplitude search, the bottom-track noise mean is subtracted, and a fixed-depth window below the surface is extracted as the "clean" reflection target. A same-shaped noise patch is independently sampled from the bottom ~10% of the same track. A synthetic noisy observation is formed as `reflection + alpha * noise`.

**Normalization.** All data is non-negative radar power. Values are log1p-transformed and rescaled using global 1st/99th percentiles computed once over the dataset, clipped to `[0, 1]`.

**Training objective.** The network predicts the noise, not the clean signal:
```
predicted_noise = model(noisy)
denoised = noisy - predicted_noise
loss = masked_SmoothL1(denoised, clean_reflection)
```
The loss is restricted to valid (finite, in-bounds) pixels via a mask. During training, patches are dynamically re-noised each epoch — a fresh noise scale is drawn per sample rather than reusing the noise baked into the stored `.npz` — so the model sees many noise realizations per clean patch.

**Current result.** The synthetic-pair supervised setup reaches **45.46 dB PSNR / 0.0033 MAE** on held-out validation patches, split by orbital track to avoid leakage between neighboring samples.

## Status & roadmap

- ✅ Implemented: end-to-end synthetic-pair supervised denoising (preprocessing → training → full-image inference → projection back to original coordinates).
- 🚧 In progress: extending to self-supervised, Noise2Self-style masked prediction, so the model can be trained and evaluated directly on real radargrams, where no clean reference target exists at all.

## Setup

```bash
conda env create -f environment.yml
conda activate pytorch
```

Key dependencies: PyTorch 2.5.1 (CUDA 12.4), NumPy, Pandas, SciPy, OpenCV, Rasterio, Cartopy, Matplotlib, Pillow, tqdm.

## Usage

```bash
# 1. Build the dataset from PDS3 RGRAM products
python data_process.py

# 2. Check dataset statistics and get recommended normalization
python check_npz_stats.py --data_dir radar_ai_dataset

# 3. Split into train/val/test by orbital track
python split_dataset.py

# 4. (optional) sanity-check the dataloader
python dataset.py --split train

# 5. Train
python train_unet.py

# 6. Run full-image inference
python infer_full_image.py --checkpoint outputs/checkpoints/best_unet.pth --split test

# 7. Project denoised patches back onto original RGRAM coordinates
python project_denoised_to_rgram.py
```

## Data sources

- SHARAD RGRAM products: [NASA PDS Geosciences Node](https://pds-geosciences.wustl.edu/)
- MOLA polar DEM: NASA PDS / USGS Astrogeology

Raw data is not included in this repository due to size and PDS licensing; scripts assume it has been downloaded locally into the paths referenced above.


Contact: janetgong3@gmail.com
