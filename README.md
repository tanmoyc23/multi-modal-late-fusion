# DeepVerse CSI Localization — two baselines

Both scripts here tackle the same basic problem: figure out where a UE
(user device) is, using wireless channel data from DeepVerse-simulated
6G scenarios, sometimes with a camera or LiDAR thrown in. They're
separate, self-contained pipelines, not two halves of one system —
`FC_AE_GPR_Reproduced.py` is the lightweight "classical" baseline, and
`Multi_Modal_Baseline_v24_embed_fusion.py` is the full deep-learning
multimodal model. Run either one on its own.

Both assume a DeepVerse scenario has already been generated on disk
(the `scenarios/<name>/param/config1.m` + raw sample files), and both
point `PROJECT_ROOT` at `/DATA/tanmoy/Documents/deepverse_localization`
by default — change that line near the top before running anywhere
else.

---

## `FC_AE_GPR_Reproduced.py`

This is a reimplementation of a published CSI-fingerprinting pipeline
(the FC-AE-GPR paper), adapted to run on DeepVerse scenario data
instead of the original dataset. It's CSI-only, no deep multimodal
stuff — just channel data in, (x, y) position out.

**What it actually does, step by step:**

1. Pulls raw CSI from DeepVerse (or a cached `.h5` copy if one already
   exists) and converts it to an Angle-Delay Profile (ADP) using the
   `V`/`F` DFT matrices from the paper's Eqs. 2–4.
2. Optionally compresses each ADP down to a 4×4 bottleneck with a small
   fully-connected autoencoder (FC-AE), trained unsupervised on a slice
   of the data. Skipped automatically if the ADP is already 4×4 — there's
   nothing left to compress.
3. Runs four localization methods side by side and compares them:
   - **FC-AE-GPR** — Gaussian Process Regression on the compressed embedding
   - **GPR** — Gaussian Process Regression directly on the raw (flattened) ADP
   - **CNN** — a small CNN fingerprinting benchmark
   - **Correlation** — nearest-neighbour matching by cosine similarity
4. Sweeps across several training-set sizes (percentages, or fixed
   absolute counts — see `TRAIN_SIZE_MODE`) and k-folds each one, so you
   get an RMSE curve, not just one number.
5. Dumps everything into a single JSON file:
   `Resultsandweights_FCAE_GPR/<experiment_name>_results.json`.

**Config worth knowing about** (all near the top of the file):

- `ADP_PRESET` — picks the antenna/ADP size (`paper4` through `paper64`).
  Bigger ADPs skip the raw-GPR row automatically once the dimensionality
  gets silly (`RAW_GPR_MAX_DIM`), and skip FC-AE below 4×4 since there's
  nothing to compress.
- `SUBCARRIER_MODE` — `'spread'` (recommended) spaces the selected
  subcarriers across the whole OFDM band instead of taking the first 8
  in a row, which is what DeepVerse's shipped config does and is honestly
  too coarse for ranging.
- `TRAIN_SIZE_MODE` — `'percent'` (matches the original paper's protocol)
  or `'absolute'` (fairer across scenarios with very different sample
  counts — DT31 only has ~7k samples vs the paper's ~72k, so the same
  percentage means very different things).
- `N_FOLDS` — the paper uses 50, this defaults to 5 to keep runtime sane.
  Just say what you ran.

Nothing in `config.m` ever gets touched — antenna shape and subcarrier
selection are overridden in memory, so this can safely share a DeepVerse
install with other experiments.

Run it with `python FC_AE_GPR_Baseline_v5.py`. CPU is fine for small ADP
presets, GPU helps once the CNN benchmark and larger GPR kernels kick in.

---

## `Multi_Modal_CADPoolFusion.py`

This is the real model — a deep multimodal localization network that
can take CSI, camera images, and LiDAR point clouds (any combination)
and predict a GPS position. Much bigger file, much more going on.

**The pipeline, roughly in order:**

1. **Cache build** — reads CSI, camera frames, and LiDAR `.pcd` files
   straight from DeepVerse / disk and writes them into one combined
   HDF5 cache (`Cache/`), so every later run just mmaps it instead of
   re-decoding images and point clouds every time. Each modality is
   checked and rebuilt independently, so changing the antenna shape
   only touches the CSI part of the cache.
2. **Feature engineering** — CSI gets turned into 12 channels (16 for a
   planar antenna array) of bounded, standardized features: raw Re/Im,
   log-magnitude, beamspace phase, subcarrier- and antenna-differential
   phase, etc. All designed to be invariant to the random carrier phase
   rotation used as a training augmentation.
3. **Per-modality encoders** — CSI, BS-position, camera, and LiDAR each
   get their own encoder, and each one outputs *two* things: a dense
   embedding (the thing that actually drives the prediction) and a
   token sequence (only consumed by the attention-refinement branch
   below).
4. **Fusion model — `GPSLocalizer`** ("DFR-Fusion"). Built so that at
   initialization it's exactly a plain dense-concatenation MLP, and every
   extra mechanism — a physics-informed beamspace "anchor," an attention
   refinement branch, a distributional (binned) output head — is bolted
   on through a zero-initialized gate. So it can never start out worse
   than the plain baseline, and only leans on the fancier stuff once
   training actually rewards it.
5. **Two fusion modes** (`FUSION_MODE`):
   - `'feature'` — everything trained jointly, end to end.
   - `'late'` — train CSI-only and camera-only models separately first
     (as frozen single-modality baselines), then combine their frozen
     predictions with `CADPool`, a small conflict-aware pooling module.
     This exists because on the DT31 scenario, joint training actually
     made things *worse* than camera-only alone — a textbook case of one
     modality drowning out the other — so freezing the experts and
     combining them downstream was the fix.
6. **Training loop** — AdamW with cosine warmup, EMA of the weights
   (checkpoints are the EMA weights, not the raw ones), annealed SmoothL1
   loss, per-modality dropout augmentation, an auxiliary single-modality
   loss term to fight modality competition, phase-rotation test-time
   augmentation, and early stopping.
7. **Evaluation + diagnostics** — full test-set metrics (mean/median/RMSE/
   p90/p95 error, 2D and 3D), an optional test-time modality ablation
   (`TEST_TIME_ABLATE`) to check whether the model is actually using a
   given sensor or just carrying it along for nothing, and training
   curve / error-CDF figures.

**What gets written out**, all under
`Resultsandweights_vtc2026_Base_Late_Fusion_WCNC/<scenario>__<modalities>__<hparams>/`:

- `best_model.pth`, `final_model.pth` — checkpoints (EMA weights)
- `training_history.csv` / `.json` — per-epoch loss, error, LR, beta
- `results.mat` / `results.json` / `results.txt` — the full test-set
  readout, in MATLAB / code / human-readable form respectively
- `curves.png` / `.pdf` — loss & LR curves
- `cdf_error.png` / `.pdf` — error CDF plot
- `ablation_<modality>.txt` — only if `TEST_TIME_ABLATE` is set

**The knobs you'll actually touch:**

- `SCENARIO_NAME` — which DeepVerse scenario to run on (`DT31`, `O1`,
  `Carla-Town01`, etc. — see `SCENARIO_REGISTRY` for what each one is).
- `USE_CSI` / `USE_CAMERA` / `USE_LIDAR` — toggle modalities on/off (also
  settable via env vars `MM_USE_CSI` / `MM_USE_CAMERA` without touching
  the file).
- `FUSION_MODE` — `'feature'` or `'late'`, see above.
- `RUN_MODE` — `'train_and_eval'` or `'eval_only'` (the latter just loads
  `best_model.pth` and re-runs the test set, no training).
- `EPOCHS`, `BATCH_SIZE`, `LR` — the usual suspects.

Needs a CUDA GPU to be practical — it'll run on CPU but the attention
refinement + distributional head make that painfully slow. Uses `bf16`
automatically on GPUs that support it, falls back to `fp16` + a grad
scaler otherwise.

---

## Requirements

Both files need: `numpy`, `h5py`, `torch`, `scikit-learn` (FC-AE-GPR
only, for the GPR), `scipy` + `matplotlib` (multimodal only, for
`.mat` saving and the figures), `Pillow` + `torchvision` (multimodal
only, for the camera path), and `tqdm` (nice to have, not required —
the FC-AE-GPR script falls back to plain print statements if it's
missing).

A working DeepVerse install with the relevant scenario folder is
assumed for both — these aren't meant to run against arbitrary data,
the cache builders expect DeepVerse's own sample/parameter layout.
