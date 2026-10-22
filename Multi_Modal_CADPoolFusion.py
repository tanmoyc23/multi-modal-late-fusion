
import os
import io
import sys
import json
import time
import math
import random
import warnings
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

import h5py
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms

import scipy.io as sio   # for .mat saving

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available() and getattr(torch.cuda, 'is_bf16_supported', lambda: False)():
    AMP_DTYPE = torch.bfloat16
    USE_GRAD_SCALER = False
else:
    AMP_DTYPE = torch.float16
    USE_GRAD_SCALER = torch.cuda.is_available()


def set_seed(seed: int = 42):
    """Seed everything so splits/results are reproducible across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(42)
warnings.filterwarnings('ignore')

print("=" * 72)
print("  MULTIMODAL GPS LOCALIZATION FROM CSI + CAMERA + LIDAR  (v5 DFR)")
print("=" * 72)
print(f"  PyTorch  : {torch.__version__}")
print(f"  Device   : {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU      : {torch.cuda.get_device_name(0)}")
    print(f"  Memory   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"  AMP dtype: {AMP_DTYPE}  (GradScaler={'on' if USE_GRAD_SCALER else 'off'})")
print("=" * 72)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# RUN_MODE: 'train_and_eval' preprocesses/trains/evaluates; 'eval_only' loads
# best_model.pth and just evaluates the test set.
RUN_MODE = 'train_and_eval'

# diagnostic only -- set to a modality name to force it to "missing" at test
# time and see if the model actually leans on it. needs MOD_DROP_P>0 for that
# modality. writes a separate ablation_<name>.txt, doesn't touch the main results.
TEST_TIME_ABLATE = None    # None | 'csi' | 'cam' | 'lidar'

# Multi-seed sweep support: USE_CSI / USE_CAMERA / SEED can be overridden via
# env vars so an external driver can launch this script as a fresh subprocess
# per (config, seed).
def _env_bool(name, default):
    v = os.environ.get(name)
    return default if v is None else (v == '1')

def _env_int(name, default):
    v = os.environ.get(name)
    return default if v is None else int(v)

USE_CSI         = _env_bool('MM_USE_CSI', True)      # CSI encoder  (the wireless input)
USE_BS_POS      = False     # BS position encoder (known infrastructure)
USE_CAMERA      = _env_bool('MM_USE_CAMERA', True)   # Camera RGB images at BS
USE_LIDAR       = False     # LiDAR point cloud at BS
USE_ALL_CAMERAS = False     # True -> use all cameras per BS, False -> camera 1 only

# Training
BATCH_SIZE     = 32
EPOCHS         = 50
LR             = 0.001      
WEIGHT_DECAY   = 1e-2
NUM_WORKERS    = 4
PATIENCE       = 45
WARMUP_EPOCHS  = 8
MIN_LR         = 1e-7
SEED           = _env_int('MM_SEED', 42)

# EMA / augmentation
EMA_DECAY      = 0.9995    
AUG_PHASE      = True      # train-time global phase rotation  h -> e^{j theta} h
AUG_NOISE_P    = 0.5       # probability of adding complex AWGN to a training CSI
AUG_SNR_DB     = (15.0, 35.0)   # SNR range (dB) for the AWGN augmentation

LOSS_BETA_START = 0.30
LOSS_BETA_END   = 0.01
LAMBDA_DIST     = 1.00     # weight of the mean-Euclidean-norm term
LAMBDA_DFL      = 0.25     # weight of the distributional (soft-bin) head loss
LAMBDA_ANCHOR   = 0.10     # deep supervision on the physics anchor

# DFR-Fusion architecture
CSI_EMBED_DIM     = 64
POS_EMBED_DIM     = 96     # BS-position dense embedding
CAM_EMBED_DIM     = 64     # camera dense embedding
LIDAR_EMBED_DIM   = 128    # LiDAR dense embedding
FUSION_DIM        = 200    # dense fusion trunk width
CSI_WIDTH         = 64     # conv width of the CSI trunk

CSI_FLATTEN_CH    = 32     # channels feeding the CSI dense-readout flatten
CAM_REDUCE_CH     = 16     # channels feeding the camera dense-readout flatten
CSI_BLOCKS        = 2      # residual blocks in the CSI trunk

USE_ATTN_REFINE   = True   # zero-init attention residual on top of the trunk
ATTN_DIM          = 64     # common token dimension for all modalities
NUM_HEADS         = 8      # attention heads (fusion + PMA)
NUM_FUSION_TOKENS = 8      # K learnable bottleneck (latent) tokens
NUM_FUSION_LAYERS = 3      # L pre-norm gated fusion layers
NUM_LIDAR_TOKENS  = 16     # M PMA seed tokens summarizing the point cloud
ATTN_LR_MULT      = 0.60   # attention/gating is the only mechanism that can
                           # down-weights an unreliable modality per sample (rel_gate). sits
                           # behind near-zero gates so needs close to base LR to get gradient

USE_DIST_HEAD     = True   # distributional (soft-binned) offset head
NUM_BINS          = 64     # bins per axis
BIN_RANGE         = 5.0    # bin span, in units of that axis' own std

LS_INIT           = 1e-2

DROPOUT           = 0.05
GRAD_CLIP         = 1.0

# per-modality dropout, symmetric across CSI/cam/lidar. an earlier version
# skewed this toward CSI, but that bias wasn't justified by the data and
# got reverted back to equal treatment for all three.
MOD_DROP_P        = {'csi': 0.15, 'cam': 0.15, 'lidar': 0.15}

# auxiliary per-modality loss: each listed modality gets an extra forced
# solo-modality pass every batch (reuses the force_drop path), added to the
# main loss at AUX_LOSS_WEIGHT. standard fix for modality competition
AUX_LOSS_MODALITIES = ['csi', 'cam']   
AUX_LOSS_WEIGHT      = 0.30


USE_CSI_CONFIDENCE_GATE = True
USE_CAM_CONFIDENCE_GATE = True

TTA_PHASE_K       = 8      # test-time phase rotations averaged (1 = off)
ARCH_TAG          = 'v5dfr'   # dense-readout / full-metric / refinement tag


SCENARIO_REGISTRY = {
    'Carla-Town01': {
        'folder': 'Carla-Town01', 'total_scenes': 2200,
        'num_bs': 5, 'num_ue_max': 200,
        'cams_per_bs': 4, 'lidars_per_bs': 1,
        'csi_std': 7.356458e-07,
        'notes': '5 BS, 100 mobile UEs, 28 GHz. Scenes 0-2199. (Primary)',
    },
    'Carla-Town05': {
        'folder': 'Carla-Town05', 'total_scenes': 4000,
        'num_bs': 3, 'num_ue_max': 200,
        'cams_per_bs': 5, 'lidars_per_bs': 1,
        'csi_std': 5.756234e-06,
        'notes': '3 BS, 100 mobile UEs, 3.5 GHz.',
    },
    'O1': {
        'folder': 'O1', 'total_scenes': 2000,
        'num_bs': 4, 'num_ue_max': 100,
        'cams_per_bs': 3, 'lidars_per_bs': 1,
        'csi_std': 7.600617e-07,
        'notes': ('4 BS, 48 mobile UEs (Unit 6-53), UEs enter from scene 41. '
                  'Unit 5 is top-view cam only (not a BS).'),
    },
    'I1': {
        'folder': 'I1', 'total_scenes': 2000,
        'num_bs': 2, 'num_ue_max': 20,
        'cams_per_bs': 0, 'lidars_per_bs': 1,
        'csi_std': 3.019425e-05,
        'notes': ('2 BS, 6 mobile UEs (humans), 3.5 GHz. Cameras on separate '
                  'units (NOT on BS) -- USE_CAMERA auto-disabled.'),
    },
    'DT1': {
        'folder': 'DT1', 'total_scenes': 2411,
        'num_bs': 1, 'num_ue_max': 5,
        'cams_per_bs': 1, 'lidars_per_bs': 1,
        'csi_std': 3.8214e-07,
        'notes': '1 BS, 1 mobile UE, 60 GHz. Digital twin of DeepSense Scenario 1.',
    },
    'DT31': {
        'folder': 'DT31', 'total_scenes': 7012,
        'num_bs': 1, 'num_ue_max': 5,
        'cams_per_bs': 1, 'lidars_per_bs': 1,
        'csi_std': 2.3917e-07,
        'notes': '1 BS, 1 mobile UE, 60 GHz. Digital twin of DeepSense Scenario 31.',
    },
}

SCENARIO_NAME = 'DT31'

assert SCENARIO_NAME in SCENARIO_REGISTRY, (
    f"Unknown scenario '{SCENARIO_NAME}'. Available: {list(SCENARIO_REGISTRY.keys())}."
)
_cfg = SCENARIO_REGISTRY[SCENARIO_NAME]
SCENARIO_FOLDER = _cfg['folder']

TOTAL_SCENES       = _cfg['total_scenes']
NUM_BS             = _cfg['num_bs']
NUM_UE_MAX         = _cfg['num_ue_max']
NUM_CAMERAS_PER_BS = _cfg['cams_per_bs']
NUM_LIDARS_PER_BS  = _cfg['lidars_per_bs']

print(f"Scenario: {SCENARIO_NAME}")
print(f"  {_cfg['notes']}")

PROJECT_ROOT  = Path("/DATA/tanmoy/Documents/deepverse_localization")
SCENARIO_PATH = str(PROJECT_ROOT / "scenarios" / SCENARIO_FOLDER)
CONFIG_PATH   = f"{SCENARIO_PATH}/param/config1.m"

IMG_H, IMG_W = 540, 960
IMG_SIZE     = (IMG_H, IMG_W)
IMG_TAG      = f"{IMG_H}_{IMG_W}"

LIDAR_POINTS = 1024
NUM_CAMERAS  = NUM_CAMERAS_PER_BS

# CSI dimensions
NUM_ANT  = 16
NUM_SUB  = 8

OFDM_TOTAL_SUBCARRIERS = 512            
OFDM_BANDWIDTH_HZ      = 0.05e9         
SUBCARRIER_MODE        = 'spread'       
if SUBCARRIER_MODE == 'spread':
    _sub_stride = OFDM_TOTAL_SUBCARRIERS // NUM_SUB
    SELECTED_SUBCARRIERS = [i * _sub_stride for i in range(NUM_SUB)]
else:
    _sub_stride = 1
    SELECTED_SUBCARRIERS = list(range(NUM_SUB))
assert max(SELECTED_SUBCARRIERS) < OFDM_TOTAL_SUBCARRIERS
SUB_TAG = f"sub{NUM_SUB}{'s' if SUBCARRIER_MODE == 'spread' else 'c'}"

# antenna array geometry -- must match dv.comm.bs_antenna.shape in config.m.
#   (16, 1) = ULA, (4, 4) = UPA (paper Table I: "16 (4 x 4)")
# antenna pattern to be changed here for comparison with FCAEGPR and contrastive learing
# perfromance check
ANT_SHAPE = (4, 4)
ANT_TAG   = f"ant{ANT_SHAPE[0]}_{ANT_SHAPE[1]}"
assert ANT_SHAPE[0] * ANT_SHAPE[1] == NUM_ANT, "ANT_SHAPE must multiply to NUM_ANT"

# ULA-vs-UPA switch. IS_UPA is True only for a genuine 2D grid (both dims >
# 1); switching back to (16,1) is a no-op on the rest of the pipeline.

ANT_ORDER = 'dim0_outer'
IS_UPA    = (ANT_SHAPE[0] > 1 and ANT_SHAPE[1] > 1)
if ANT_ORDER == 'dim0_outer':
    ANT_ROWS, ANT_COLS = ANT_SHAPE[0], ANT_SHAPE[1]
else:
    ANT_ROWS, ANT_COLS = ANT_SHAPE[1], ANT_SHAPE[0]

CSI_STD = float(_cfg['csi_std'])   

# 12 CSI channels for a ULA, 14 for a UPA (extra 2 replace the single
# antenna-differential pair). all bounded cos/sin/standardized, computed at
# load time from raw Re/Im -- no cache rebuild needed if only these change.
#   ch 0-1  : Re(h)/csi_std, Im(h)/csi_std       -- unit-scale linear field
#   ch 2    : standardized log10 |h|             -- path-loss / range cue
#   ch 3-4  : cos(phase), sin(phase)             -- bounded, wrap-free phase
#   ch 5    : standardized log10 |FFT2(h)|       -- angle-delay / beamspace
#   ch 6-7  : cos/sin of angle(A)                -- sub-bin AoA/delay that |A| drops
#   ch 8-9  : cos/sin of angle(h[a,k+1] conj(h[a,k]))  -- subcarrier-diff phase, ~ delay/range
#   -- ULA (12 channels total) --
#   ch 10-11: cos/sin of angle(h[a+1,k] conj(h[a,k]))  -- antenna-diff phase, ~ sin(AoA)
#   -- UPA (14 channels total) --
#   ch 10-11: cos/sin of angle(h[row,col+1,k] conj(h[row,col,k]))  -- column diff, azimuth
#   ch 12-13: cos/sin of angle(h[row+1,col,k] conj(h[row,col,k]))  -- row diff, elevation
#   both differentials computed on the true 2D grid (per ANT_ORDER) so phase
#   never mixes across a row/column boundary.
CSI_FEAT_CH = 14 if IS_UPA else 12

print(f"  IMG            = {IMG_H}x{IMG_W}  (paper: 540x960)")
print(f"  ANT_SHAPE      = {ANT_SHAPE}  ({ANT_TAG})  "
      f"[{'UPA' if IS_UPA else 'ULA'}, order={ANT_ORDER}, rows={ANT_ROWS}, cols={ANT_COLS}]")
print(f"  CSI_STD (per-scenario) = {CSI_STD:.6e}")
print(f"  CSI_FEAT_CH            = {CSI_FEAT_CH}  "
      f"(v6: {'UPA 2D row/col differential phase' if IS_UPA else 'ULA 1D differential phase (unchanged from v5)'})")

NUM_ANCHOR_STATS = 14 if IS_UPA else 12

# Model output
OUT_DIM = 3

# Data split
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# (test is the rest -- so 0.15)

# fusion mode.
# 'feature' -- both encoders trained together end-to-end, concatenated into
#              the FUSION_DIM trunk.
# 'late'    -- CSI-only and camera-only trained separately first, then
#              frozen and combined by CADPool below.
FUSION_MODE = 'late'          # 'late' | 'feature'

# CADPool (Conflict-Aware Distributional Pooling) hyperparameters
LATE_HIDDEN       = 96      # width of the tiny descriptor MLP
LATE_DROPOUT      = 0.05    # dropout inside the combiner MLP
LATE_MOD_DROP_P   = 0.15    # p(force the combiner to use one expert alone)
                            # per batch/branch, train only -- teaches the
                            # combiner a sane single-expert fallback
LATE_GATE_CLAMP   = 3.0     # |log-domain| clamp on the learned pooling
                            # exponents; exp(+-3) bounds how far the learned
                            # weight can move the classical precision weight
LATE_CORR_SCALE   = 0.10    # scale on the zero-init residual correction, in
                            # isotropic offset units (~1.06 m of authority);
                            # starts at exactly zero (CADPool._zero_init) so
                            # epoch 0 reproduces the classical estimator
LATE_EMB_HIDDEN   = 32      # width each expert's penultimate embedding is
                            # compressed before the gating MLP -- kept small since CADPool
                            # fits on only a few hundred to ~1k samples (LATE_FIT_ON)

# Cache / data locations
CACHE_DIR = PROJECT_ROOT / "Cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_SC = SCENARIO_FOLDER.lower().replace('-', '_')

HDF5_FILE     = CACHE_DIR / f"{_SC}_img{IMG_TAG}_lid{LIDAR_POINTS}_{ANT_TAG}_{SUB_TAG}_u8.h5"
PROGRESS_FILE = CACHE_DIR / f"{_SC}_img{IMG_TAG}_{ANT_TAG}_{SUB_TAG}_progress.json"

VALID_SAMPLES_FILE = CACHE_DIR / f"{_SC}_valid_samples.json"

LEGACY_H5 = CACHE_DIR / f"{_SC}_img64_lid{LIDAR_POINTS}_nozip.h5"

print(f"  CACHE_DIR      = {CACHE_DIR}")
print(f"  HDF5_FILE      = {HDF5_FILE.name}")
print(f"  SUBCARRIERS    = {NUM_SUB}/{OFDM_TOTAL_SUBCARRIERS} '{SUBCARRIER_MODE}' "
      f"stride={_sub_stride}  indices={SELECTED_SUBCARRIERS}")
print(f"  VALID_SAMPLES  = {VALID_SAMPLES_FILE.name}")

assert any([USE_CSI, USE_BS_POS, USE_CAMERA, USE_LIDAR]), "Enable at least one modality!"
assert RUN_MODE in ('train_and_eval', 'eval_only'), \
    "RUN_MODE must be 'train_and_eval' or 'eval_only'"
assert 0.0 < LOSS_BETA_END <= LOSS_BETA_START, "Require 0 < LOSS_BETA_END <= LOSS_BETA_START"
assert NUM_BINS >= 8 and BIN_RANGE > 0, "Distributional head needs a sane bin grid"
if USE_CAMERA and NUM_CAMERAS_PER_BS == 0:
    print(f"WARNING: {SCENARIO_NAME} has no cameras collocated with BS. Disabling USE_CAMERA.")
    USE_CAMERA = False


def _hp_tag() -> str:
    """Hyperparameter half of the experiment folder name.

    Factored out so the late-fusion checkpoint lookup can reconstruct the
    CSI-only/camera-only folder names from the same source of truth --
    otherwise late fusion could silently pick up the wrong base checkpoint.
    """
    return (
        f"{ARCH_TAG}_bs{BATCH_SIZE}_lr{LR:.0e}_ep{EPOCHS}"
        f"_pat{PATIENCE}_wu{WARMUP_EPOCHS}"
        f"_fd{FUSION_DIM}_ce{CSI_EMBED_DIM}"
        f"_ad{ATTN_DIM}_ft{NUM_FUSION_TOKENS}x{NUM_FUSION_LAYERS}"
        f"_nb{NUM_BINS if USE_DIST_HEAD else 0}"
        f"_do{DROPOUT}_seed{SEED}"
    )


def _csi_mod_tag() -> str:
    return f"csi-{ANT_TAG}-{SUB_TAG}"


def _cam_mod_tag() -> str:
    return f"cam{NUM_CAMERAS}" if USE_ALL_CAMERAS else "cam1"


IS_LATE_FUSION = (FUSION_MODE == 'late') and USE_CSI and USE_CAMERA


def get_experiment_name() -> str:
    mod = []
    if USE_CSI:    mod.append(_csi_mod_tag())    
    if USE_BS_POS: mod.append("bspos")
    if USE_CAMERA:
        mod.append(_cam_mod_tag())
    if USE_LIDAR:  mod.append("lidar")
    if not mod:    mod.append("none")
    mod_tag = "_".join(mod)
    hp_tag = _hp_tag()
    if IS_LATE_FUSION:
        hp_tag += "_late"
    return f"{SCENARIO_NAME}__{mod_tag}__{hp_tag}"



EXPERIMENT_NAME = get_experiment_name()
RESULTS_ROOT    = Path("Resultsandweights_vtc2026_Base_Late_Fusion_WCNC")
RESULTS_ROOT.mkdir(exist_ok=True)
OUTPUT_DIR      = RESULTS_ROOT / EXPERIMENT_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_base_checkpoint(kind: str) -> Path:
    """Locate the frozen base checkpoint for late fusion. kind: 'csi'|'cam'.

    Rebuilds the folder name from the current config via _hp_tag(), so it
    finds the CSI-only/camera-only run at this exact geometry, subcarrier
    count, seed and architecture -- and raises with the exact fix instead of
    silently falling back to something stale or wrong.
    """
    mod_tag = _csi_mod_tag() if kind == 'csi' else _cam_mod_tag()
    name    = f"{SCENARIO_NAME}__{mod_tag}__{_hp_tag()}"
    ckpt    = RESULTS_ROOT / name / "best_model.pth"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"\n[late fusion] Missing the frozen {kind.upper()} base model.\n"
            f"  expected: {ckpt}\n\n"
            f"Late fusion consumes two ALREADY-TRAINED single-modality models.\n"
            f"To produce this one, run this script once with:\n"
            f"    FUSION_MODE   = 'late'   (ignored for single-modality runs)\n"
            f"    USE_CSI       = {kind == 'csi'}\n"
            f"    USE_CAMERA    = {kind == 'cam'}\n"
            f"and everything else (ANT_SHAPE, NUM_SUB, FUSION_DIM, SEED, ...) \n"
            f"left EXACTLY as it is now, then re-run this fusion configuration.\n")
    return ckpt


def _load_frozen_base(kind: str, device):
    """Build a single-modality GPSLocalizer, load its EMA weights, freeze it.

    Every architectural field that affects the fused computation is checked
    against the checkpoint's own saved config -- a base model trained with a
    different bin grid or output dim can't be pooled with one that wasn't.
    """
    ckpt_path = find_base_checkpoint(kind)
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:                       # older torch without weights_only
        ckpt = torch.load(ckpt_path, map_location=device)

    # checkpoint dict is flat (_checkpoint_payload's fields sit directly on
    # ckpt, there's no nested 'config' key)
    cfg = ckpt if isinstance(ckpt, dict) else {}
    for key, want in (('use_dist_head', True),
                      ('num_bins',      NUM_BINS),
                      ('out_dim',       OUT_DIM)):
        if key in cfg and cfg[key] != want:
            raise ValueError(
                f"[late fusion] {kind} checkpoint {ckpt_path} was trained with "
                f"{key}={cfg[key]!r} but this run needs {key}={want!r}. CADPool "
                f"pools the two experts on a SHARED bin grid, so both base "
                f"models must agree on it. Retrain that base model at the "
                f"current settings.")

    # NUM_SUB/ANT_SHAPE mismatch used to surface as an opaque PyTorch
    # "size mismatch for csi_encoder.pos_emb" dump -- checked proactively
    # here with a message that names the actual cause.
    if kind == 'csi':
        if 'num_ant' in cfg and 'num_sub' in cfg:
            if (cfg['num_ant'], cfg['num_sub']) != (NUM_ANT, NUM_SUB):
                raise ValueError(
                    f"[late fusion] {kind} checkpoint {ckpt_path} was trained "
                    f"with NUM_ANT={cfg['num_ant']}, NUM_SUB={cfg['num_sub']} "
                    f"but this run has NUM_ANT={NUM_ANT}, NUM_SUB={NUM_SUB}.\n"
                    f"This is almost certainly the folder-naming bug fixed in "
                    f"this version: an older checkpoint from a different "
                    f"subcarrier/antenna config sat at this path and was "
                    f"loaded by mistake. Either set NUM_SUB={cfg['num_sub']} "
                    f"to match what is actually saved, or retrain CSI-only at "
                    f"the current NUM_SUB={NUM_SUB} -- the new folder naming "
                    f"(includes SUB_TAG) means it will no longer collide with "
                    f"a checkpoint trained at a different subcarrier count.")
        elif 'model_state_dict' in ckpt and 'csi_encoder.pos_emb' in ckpt['model_state_dict']:
            # older checkpoint, saved before num_ant/num_sub were recorded --
            # check the raw tensor shape instead of letting load_state_dict
            # fail with an unrelated wall of key names
            saved_len = ckpt['model_state_dict']['csi_encoder.pos_emb'].shape[1]
            want_len  = NUM_ANT * NUM_SUB
            if saved_len != want_len:
                saved_sub = saved_len // max(NUM_ANT, 1)
                raise ValueError(
                    f"[late fusion] {kind} checkpoint {ckpt_path} has "
                    f"pos_emb length {saved_len} (implies NUM_ANT*NUM_SUB, "
                    f"likely NUM_SUB={saved_sub} at NUM_ANT={NUM_ANT}) but "
                    f"this run expects {want_len} (NUM_ANT={NUM_ANT}, "
                    f"NUM_SUB={NUM_SUB}). This checkpoint predates this fix "
                    f"and didn't record its config explicitly, but the shapes "
                    f"disagree -- either set NUM_SUB={saved_sub} to match it, "
                    f"or retrain CSI-only at the current settings.")

    if cfg.get('use_dist_head') is False:
        raise ValueError(
            f"[late fusion] {kind} checkpoint has USE_DIST_HEAD=False, so it "
            f"emits no per-axis distribution. CADPool fuses distributions, not "
            f"just points -- retrain that base model with USE_DIST_HEAD=True.")

    m = GPSLocalizer(
        use_csi=(kind == 'csi'), use_bs_pos=USE_BS_POS,
        use_camera=(kind == 'cam'), use_lidar=False,
        use_all_cameras=USE_ALL_CAMERAS, num_cameras=NUM_CAMERAS,
        csi_embed_dim=CSI_EMBED_DIM, pos_embed_dim=POS_EMBED_DIM,
        cam_embed_dim=CAM_EMBED_DIM, lidar_embed_dim=LIDAR_EMBED_DIM,
        fusion_dim=FUSION_DIM, dropout=DROPOUT,
        csi_width=CSI_WIDTH, csi_blocks=CSI_BLOCKS,
        csi_flatten_ch=CSI_FLATTEN_CH, cam_reduce_ch=CAM_REDUCE_CH,
        num_ant=NUM_ANT, num_sub=NUM_SUB, csi_in_ch=CSI_FEAT_CH, out_dim=OUT_DIM,
        img_hw=IMG_SIZE,
        use_attn_refine=USE_ATTN_REFINE, attn_dim=ATTN_DIM, num_heads=NUM_HEADS,
        num_fusion_tokens=NUM_FUSION_TOKENS, num_fusion_layers=NUM_FUSION_LAYERS,
        num_lidar_tokens=NUM_LIDAR_TOKENS,
        use_dist_head=True, num_bins=NUM_BINS, bin_range=BIN_RANGE,
        axis_scale=AXIS_SCALE, mod_drop_p=MOD_DROP_P, ls_init=LS_INIT,
        use_csi_confidence_gate=USE_CSI_CONFIDENCE_GATE,
        use_cam_confidence_gate=USE_CAM_CONFIDENCE_GATE,
        is_upa=IS_UPA, ant_rows=ANT_ROWS, ant_cols=ANT_COLS,
    ).to(device)

    missing, unexpected = m.load_state_dict(ckpt['model_state_dict'], strict=False)
    missing    = [k for k in missing    if 'missing_emb' not in k]
    unexpected = [k for k in unexpected if 'missing_emb' not in k]
    if missing or unexpected:
        raise RuntimeError(
            f"[late fusion] {kind} checkpoint does not match the architecture "
            f"this run builds.\n  missing keys   : {missing[:6]}\n"
            f"  unexpected keys: {unexpected[:6]}\n"
            f"This means the base model was trained at different architectural "
            f"settings. Retrain it at the current settings.")
    for p in m.parameters():
        p.requires_grad_(False)
    m.eval()
    print(f"[OK] frozen {kind} expert loaded  <- {ckpt_path.parent.name}"
          f"  (val {ckpt.get('val_err', float('nan')):.4f} m @ epoch {ckpt.get('epoch','?')})")
    return m


print("\n" + "=" * 72)
print("CONFIGURATION")
print("=" * 72)
print(f"  RUN_MODE         : {RUN_MODE}")
print(f"  ARCH             : {ARCH_TAG}  (dense full-resolution readout + "
      f"zero-init attention refinement)")
print(f"  USE_CSI          : {USE_CSI}")
print(f"  USE_BS_POS       : {USE_BS_POS}")
print(f"  USE_CAMERA       : {USE_CAMERA}  (all={USE_ALL_CAMERAS}, {NUM_CAMERAS_PER_BS} cams/BS)")
print(f"  USE_LIDAR        : {USE_LIDAR}")
print(f"  SCENARIO         : {SCENARIO_NAME} ({SCENARIO_FOLDER})")
print(f"  SCENES/BS/UEs    : {TOTAL_SCENES} / {NUM_BS} / {NUM_UE_MAX}")
print(f"  CSI_STD          : {CSI_STD:.6e}  (per-scenario)")
print(f"  CSI features     : {CSI_FEAT_CH} ch  |  trunk {CSI_WIDTH}ch x {CSI_BLOCKS} blocks")
print(f"  BATCH/EPOCHS     : {BATCH_SIZE} / {EPOCHS}")
print(f"  LR / WD          : {LR} / {WEIGHT_DECAY}  (attention LR x{ATTN_LR_MULT})")
print(f"  PATIENCE/WARMUP  : {PATIENCE} / {WARMUP_EPOCHS}")
print(f"  LOSS             : SmoothL1(beta {LOSS_BETA_START} -> {LOSS_BETA_END}) "
      f"+ {LAMBDA_DIST}*mean||.||2")
print(f"                     + {LAMBDA_DFL}*DFL + {LAMBDA_ANCHOR}*anchor")
print(f"  DENSE FUSION     : dim {FUSION_DIM}  <- csi {CSI_EMBED_DIM} | bs {POS_EMBED_DIM} "
      f"| cam {CAM_EMBED_DIM} | lidar {LIDAR_EMBED_DIM}")
print(f"  ATTN REFINE      : {USE_ATTN_REFINE}  ({NUM_FUSION_TOKENS} tokens x "
      f"{NUM_FUSION_LAYERS} layers, dim {ATTN_DIM}, {NUM_HEADS} heads, LayerScale init 0)")
print(f"  DIST HEAD        : {USE_DIST_HEAD}  ({NUM_BINS} bins/axis, +-{BIN_RANGE} axis-std)")
print(f"  Mod. dropout     : "
      f"{', '.join(f'{k}={v}' for k, v in MOD_DROP_P.items()) if isinstance(MOD_DROP_P, dict) else f'p={MOD_DROP_P}'}")
print(f"  LayerScale init  : {LS_INIT}  (attention LR x{ATTN_LR_MULT})")
print(f"  Aux modality loss: modalities={AUX_LOSS_MODALITIES}  "
      f"weight={AUX_LOSS_WEIGHT}  (deterministic every-batch solo loss, "
      f"see MOD_DROP_P comment)")
print(f"  CSI confidence gt: {USE_CSI_CONFIDENCE_GATE}  "
      f"(GMU-style multiplicative pre-trunk gate, Arevalo et al. ICLR 2017 WS)")
print(f"  Cam confidence gt: {USE_CAM_CONFIDENCE_GATE}  (symmetric mirror of "
      f"the CSI gate -- both computed from the same [csi;cam] context)")
print(f"  EMA decay        : {EMA_DECAY}  (bias-corrected)")
print(f"  Augmentation     : phase-rot={AUG_PHASE}, AWGN p={AUG_NOISE_P} "
      f"SNR {AUG_SNR_DB[0]:.0f}-{AUG_SNR_DB[1]:.0f} dB")
print(f"  TTA (test)       : {TTA_PHASE_K} phase rotations averaged")
print(f"  HDF5 cache       : {HDF5_FILE}")
print(f"  Valid samples    : {VALID_SAMPLES_FILE}")
print(f"  OUTPUT_DIR       : {OUTPUT_DIR}")
print("=" * 72)

ENSURE_CSI    = True     # guarantee a valid 'csi' dataset in HDF5_FILE
ENSURE_IMAGES = True     # guarantee a valid 'images' dataset in HDF5_FILE
ENSURE_LIDAR  = True     # guarantee a valid 'lidar' dataset in HDF5_FILE

FORCE_REBUILD_CSI    = False   # True -> ignore any valid/harvestable csi, regen via DeepVerse
FORCE_REBUILD_IMAGES = False   # True -> ignore any valid/harvestable images, re-read PNGs
FORCE_REBUILD_LIDAR  = False   # True -> ignore any valid/harvestable lidar, re-parse PCDs
FORCE_FULL_REBUILD   = False   # convenience: True forces all three above

IMG_EXTS     = ('.png', '.jpg', '.jpeg')   
SCENE_OFFSET = 0                            # DT31 frames are named 0.png .. 7011.png


def _find_frame(bs_unit, cam_id, scene):
    """Locate one camera frame on disk, tolerant of extension/zero-padding."""
    base = f"{SCENARIO_PATH}/RGB_images/unit{bs_unit}_cam{cam_id}"
    n = scene + SCENE_OFFSET
    for stem in (f"{n}", f"{n:05d}", f"{n:06d}"):
        for ext in IMG_EXTS:
            p = f"{base}/{stem}{ext}"
            if os.path.exists(p):
                return p
    return None


def _load_frame(p):
    """PNG/JPG on disk -> (3, IMG_H, IMG_W) uint8. Normalization happens at
    __getitem__ time, not here."""
    img = Image.open(p).convert('RGB').resize((IMG_W, IMG_H), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)


def _read_pcd_xyz(path):
    with open(path, 'rb') as fb:
        header = {}
        while True:
            line = fb.readline().decode('ascii', errors='ignore').strip()
            parts = line.split()
            if parts:
                header[parts[0].upper()] = parts[1:]
            if line.upper().startswith('DATA'):
                data_type = parts[1].lower() if len(parts) > 1 else 'ascii'
                break

        fields = header.get('FIELDS', ['x', 'y', 'z'])
        sizes  = [int(s) for s in header.get('SIZE',  ['4'] * len(fields))]
        types  = header.get('TYPE',  ['F'] * len(fields))
        counts = [int(c) for c in header.get('COUNT', ['1'] * len(fields))]
        n_pts  = int(header.get('POINTS', header.get('WIDTH', ['0']))[0])

        if not all(a in fields for a in ('x', 'y', 'z')):
            raise ValueError(f"{path}: no x/y/z in FIELDS={fields}")

        if data_type == 'ascii':
            body = fb.read().decode('ascii', errors='ignore')
            rows = [r.split() for r in body.strip().splitlines() if r.strip()]
            arr = np.array(rows, dtype=np.float32)
            idx = [fields.index(a) for a in ('x', 'y', 'z')]
            return arr[:, idx]

        if data_type == 'binary':
            tmap = {('F', 4): 'f4', ('F', 8): 'f8',
                   ('U', 1): 'u1', ('U', 2): 'u2', ('U', 4): 'u4',
                   ('I', 1): 'i1', ('I', 2): 'i2', ('I', 4): 'i4'}
            dt_fields = []
            for name, t, s, c in zip(fields, types, sizes, counts):
                npt = tmap.get((t, s), 'f4')
                for k in range(c):
                    dt_fields.append((f"{name}_{k}" if c > 1 else name, npt))
            struct_dt = np.dtype(dt_fields)
            raw = fb.read(n_pts * struct_dt.itemsize)
            arr = np.frombuffer(raw, dtype=struct_dt, count=n_pts)
            return np.stack([arr['x'].astype(np.float32),
                             arr['y'].astype(np.float32),
                             arr['z'].astype(np.float32)], axis=1)

        raise ValueError(f"{path}: unsupported DATA type '{data_type}' "
                         f"(binary_compressed needs open3d)")


# one validity check per modality: dataset exists, right row count, matches
# the params that affect its content (ant shape for csi, resolution for
# images), and isn't degenerate (non-zero std).
def _csi_valid(f, N):
    # also checks subcarrier count/selection, not just sample count and
    # antenna shape -- otherwise a cache with different NUM_SUB could get
    # harvested as a match with the wrong frequency content.
    return ('csi' in f and f['csi'].shape[0] == N
           and f['csi'].shape[-1] == NUM_SUB
           and list(f.attrs.get('ant_shape', [])) == list(ANT_SHAPE)
           and list(f.attrs.get('selected_subcarriers', [])) == list(SELECTED_SUBCARRIERS)
           and float(np.abs(f['csi'][:256]).std()) > 0.0)


def _images_valid(f, N):
    return ('images' in f and f['images'].shape[0] == N
           and int(f.attrs.get('img_h', -1)) == IMG_H
           and int(f.attrs.get('img_w', -1)) == IMG_W
           and float(f['images'][:64].astype(np.float32).std()) > 1.0)


def _lidar_valid(f, N):
    return ('lidar' in f and f['lidar'].shape[0] == N
           and int(f.attrs.get('lidar_points', -1)) == LIDAR_POINTS
           and float(f['lidar'][:256].astype(np.float32).std()) > 1e-6)


def _find_harvest_source(predicate, N, exclude=None):
    """Scan every cache file for this scenario for one whose dataset
    satisfies `predicate`. Broad glob so it also finds the legacy
    dt31_img64_lid1024_nozip.h5, which has real 16x1 CSI and real LiDAR
    under an older naming convention."""
    for p in sorted(CACHE_DIR.glob(f"{_SC}_img*.h5")):
        if exclude is not None and p.resolve() == exclude.resolve():
            continue
        if not p.exists():
            continue
        try:
            with h5py.File(p, 'r') as f:
                if predicate(f, N):
                    return p
        except Exception:
            continue
    return None


def _write_full_dataset(f, key, data, chunks):
    """(Re)create `key` in the open file `f` with `data`, replacing any
    existing dataset of the same name first."""
    if key in f:
        del f[key]
    f.create_dataset(key, data=data, dtype=data.dtype, chunks=chunks)


def _backfill_legacy_attrs():
    """One-time, idempotent. LEGACY_H5 predates the ant_shape/img_h/img_w
    attribute convention the validity predicates rely on -- without this,
    its genuinely-valid 16x1 CSI and LiDAR would never be recognized as
    harvestable. Backfills ant_shape=[16,1] (known from how the file was
    built) and deliberately skips img_h/img_w, since its 64x64 images must
    never be harvested at any resolution.

    Also predates subcarrier tracking: it was built before any override
    existed, so its CSI is the config files' literal default (8 contiguous
    subcarriers, indices 0-7). Backfilling that exact fact (not the current
    SELECTED_SUBCARRIERS) keeps it a valid harvest source for a genuine
    NUM_SUB=8/'contiguous' request."""
    if not LEGACY_H5.exists():
        return
    try:
        with h5py.File(LEGACY_H5, 'r') as f:
            if 'ant_shape' in f.attrs and 'selected_subcarriers' in f.attrs:
                return
        with h5py.File(LEGACY_H5, 'a') as f:
            if 'ant_shape' not in f.attrs:
                f.attrs['ant_shape'] = [16, 1]
            if 'selected_subcarriers' not in f.attrs:
                f.attrs['selected_subcarriers'] = [0, 1, 2, 3, 4, 5, 6, 7]
                f.attrs['subcarrier_mode'] = 'contiguous'
            if 'lidar_points' not in f.attrs and 'lidar' in f:
                f.attrs['lidar_points'] = f['lidar'].shape[1]
        print(f"[OK] Backfilled legacy attrs on {LEGACY_H5.name} "
              f"(ant_shape=[16,1], selected_subcarriers=[0..7] contiguous)")
    except Exception as e:
        print(f"[!] Could not backfill legacy attrs on {LEGACY_H5.name}: {e}")


def _set_nested(params, path_options, value, what):
    """Try each candidate key-path in turn, set the first that exists in
    `params`. DeepVerse's internal key naming isn't inspectable from here,
    so trying a short list of plausible paths and reporting which matched
    is safer than guessing one."""
    applied = []
    for path in path_options:
        node, ok = params, True
        for k in path[:-1]:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                ok = False; break
        if ok and isinstance(node, dict):
            node[path[-1]] = value
            applied.append('.'.join(path))
    print(f"    set {what} via: {', '.join(applied) if applied else '(NONE -- relying on config.m)'}")
    return applied


# DeepVerse init (CSI rebuild only -- never called for images or lidar)
def _init_deepverse():
    try:
        import deepverse as dv
        from deepverse import ParameterManager
    except ImportError as e:
        raise RuntimeError(
            "DeepVerse is required to rebuild CSI (no valid/harvestable "
            "csi found) but is not installed."
        ) from e

    cfg = CONFIG_PATH
    if not os.path.exists(cfg):
        for alt in ('config.m', 'config1.m'):
            alt_path = f"{SCENARIO_PATH}/param/{alt}"
            if os.path.exists(alt_path):
                cfg = alt_path
                break
    print(f"  Config file : {cfg}")
    print(f"  NOTE: dv.comm.bs_antenna.shape in that file MUST equal "
          f"ANT_SHAPE={ANT_SHAPE} for the CSI to match what this script expects.")

    pm = ParameterManager(cfg)

    # dv.dataset_folder in config.m is relative ('scenarios') by design --
    # override to absolute so DeepVerse resolves it regardless of CWD.
    pm.params["dataset_folder"] = str(PROJECT_ROOT / "scenarios")
    _yaml_check = PROJECT_ROOT / "scenarios" / SCENARIO_FOLDER / "param" / "scenario.yaml"
    print(f"  CWD (for reference) : {os.getcwd()}")
    print(f"  dataset_folder (abs): {pm.params['dataset_folder']}")
    assert _yaml_check.exists(), (
        f"scenario.yaml not found at {_yaml_check}. "
        f"Check PROJECT_ROOT and SCENARIO_FOLDER above."
    )

    pm.params["scenes"]          = list(range(TOTAL_SCENES))
    pm.params["basestations"]    = list(range(1, NUM_BS + 1))
    pm.params["radar"]["enable"] = False
    pm.params["comm"]["enable"]  = True
    pm.params["camera"]          = True
    pm.params["dataset_folder"]  = str(PROJECT_ROOT/"scenarios")
    pm.params["lidar"]           = True
    pm.params["position"]        = True
    pm.params["camera_id"] = [
        f"unit{bs}_cam{cam}"
        for bs in range(1, NUM_BS + 1)
        for cam in range(1, NUM_CAMERAS_PER_BS + 1)
    ]
    pm.params["lidar_id"] = [
        f"unit{bs}_lidar{lid}"
        for bs in range(1, NUM_BS + 1)
        for lid in range(1, NUM_LIDARS_PER_BS + 1)
    ]

    # Override the subcarrier selection in memory only -- config.m/config1.m
    # on disk are never touched, same convention as bs_antenna.shape above.
    print("  overriding OFDM.selected_subcarriers in memory (config file untouched):")
    _set_nested(pm.params,
                [["comm", "OFDM", "selected_subcarriers"],
                 ["comm", "ofdm", "selected_subcarriers"],
                 ["comm", "OFDM_selected_subcarriers"]],
                list(SELECTED_SUBCARRIERS),
                f"OFDM.selected_subcarriers ({NUM_SUB} bins, '{SUBCARRIER_MODE}')")

    _stdout, _stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        dv_dataset = dv.Dataset(pm)
    finally:
        sys.stdout, sys.stderr = _stdout, _stderr
    print("[OK] DeepVerse ready")
    return dv_dataset


def _scan_valid_samples():
    """Return {'samples','bs_positions','ue_positions'}. Loaded as-is from
    VALID_SAMPLES_FILE if present -- it can't be rebuilt without DeepVerse
    and the raw scenario data, so it's never regenerated once it exists."""
    if VALID_SAMPLES_FILE.exists():
        with open(VALID_SAMPLES_FILE) as f:
            vdata = json.load(f)
        print(f"[OK] Loaded {len(vdata['samples']):,} valid samples "
              f"from {VALID_SAMPLES_FILE.name}")
        return vdata

    print("[..] valid_samples.json missing -- scanning DeepVerse (one-time, slow)")
    dv_dataset = _init_deepverse()
    valid_samples, bs_positions, ue_positions = [], {}, {}
    for scene in tqdm(range(TOTAL_SCENES), desc="Scenes"):
        for bs in range(NUM_BS):
            bs_key = f"{scene}_{bs}"
            if bs_key not in bs_positions:
                try:
                    p = dv_dataset.get_sample('loc-bs', index=scene, bs_idx=bs)
                    if p is not None:
                        bs_positions[bs_key] = [float(x) for x in p]
                except Exception:
                    pass
            for ue in range(NUM_UE_MAX):
                try:
                    comm = dv_dataset.get_sample('comm-ue', index=scene, bs_idx=bs, ue_idx=ue)
                except Exception:
                    continue
                if not (comm and hasattr(comm, 'coeffs') and comm.coeffs is not None):
                    continue
                valid_samples.append([scene, bs, ue])
                ue_key = f"{scene}_{ue}"
                if ue_key not in ue_positions:
                    try:
                        p = dv_dataset.get_sample('loc-ue', index=scene, ue_idx=ue)
                        if p is not None:
                            ue_positions[ue_key] = [float(x) for x in p]
                    except Exception:
                        pass

    vdata = {'samples': valid_samples, 'bs_positions': bs_positions,
             'ue_positions': ue_positions}
    with open(VALID_SAMPLES_FILE, 'w') as f:
        json.dump(vdata, f)
    print(f"[OK] Found {len(valid_samples):,} valid samples -> {VALID_SAMPLES_FILE.name}")
    return vdata


# Per-modality rebuilders -- each assumes an open h5py file handle `f` and
# (re)creates exactly one dataset from its original source.
def _rebuild_csi(f, samples, N):
    if 'csi' in f:
        del f['csi']
    f.create_dataset('csi', shape=(N, 2, NUM_ANT, NUM_SUB), dtype='float32',
                     chunks=(min(500, N), 2, NUM_ANT, NUM_SUB))
    dv_dataset = _init_deepverse()
    for idx in tqdm(range(N), desc="  csi (DeepVerse)"):
        scene, bs, ue = samples[idx]
        comm = dv_dataset.get_sample('comm-ue', index=scene, bs_idx=bs, ue_idx=ue)
        csi = comm.coeffs.reshape(NUM_ANT, NUM_SUB)
        f['csi'][idx] = np.stack([np.real(csi), np.imag(csi)], axis=0).astype(np.float32)
        if (idx + 1) % 1000 == 0:
            f.flush()
    f.attrs['ant_shape'] = list(ANT_SHAPE)
    f.attrs['selected_subcarriers'] = list(SELECTED_SUBCARRIERS)
    f.attrs['subcarrier_mode'] = SUBCARRIER_MODE


def _rebuild_images(f, samples, N, n_cams):
    if 'images' in f:
        del f['images']
    f.create_dataset('images', shape=(N, n_cams, 3, IMG_H, IMG_W), dtype='uint8',
                     chunks=(1, n_cams, 3, IMG_H, IMG_W))
    for idx in tqdm(range(N), desc="  images (PNG)"):
        scene, bs, ue = samples[idx]
        bs_unit = bs + 1
        frames = []
        for cam_id in range(1, NUM_CAMERAS_PER_BS + 1):
            p = _find_frame(bs_unit, cam_id, scene)
            if p is None:
                raise FileNotFoundError(
                    f"No camera frame for unit{bs_unit}_cam{cam_id}, scene "
                    f"{scene} in {SCENARIO_PATH}/RGB_images/  (tried {IMG_EXTS})")
            frames.append(_load_frame(p))
        f['images'][idx] = np.stack(frames, axis=0)
        if (idx + 1) % 1000 == 0:
            f.flush()
    f.attrs['img_h'] = IMG_H
    f.attrs['img_w'] = IMG_W


def _rebuild_lidar(f, samples, N):
    if 'lidar' in f:
        del f['lidar']
    f.create_dataset('lidar', shape=(N, LIDAR_POINTS, 3), dtype='float32',
                     chunks=(min(100, N), LIDAR_POINTS, 3))
    n_filled = n_missing = n_error = 0
    for idx in tqdm(range(N), desc="  lidar (PCD)"):
        scene, bs, ue = samples[idx]
        bs_unit = bs + 1
        lp = f"{SCENARIO_PATH}/lidar/unit{bs_unit}_lidar1/{scene}.pcd"
        pts_out = np.zeros((LIDAR_POINTS, 3), np.float32)
        if os.path.exists(lp):
            try:
                raw = _read_pcd_xyz(lp)
                if len(raw) > 0:
                    # same normalization convention used throughout this file
                    raw = (raw - raw.mean(0)) / 100.0
                    if len(raw) >= LIDAR_POINTS:
                        sel = np.random.choice(len(raw), LIDAR_POINTS, replace=False)
                        pts_out = raw[sel]
                    else:
                        pts_out[:len(raw)] = raw
                    n_filled += 1
            except Exception as e:
                n_error += 1
                if n_error <= 5:
                    print(f"  [!] scene {scene}: {e}")
        else:
            n_missing += 1
        f['lidar'][idx] = pts_out
        if (idx + 1) % 1000 == 0:
            f.flush()
    f.attrs['lidar_points'] = LIDAR_POINTS
    print(f"  lidar: filled={n_filled:,}  missing_file={n_missing:,}  "
         f"parse_error={n_error:,}  / {N:,}")


def _verify_cache(path, n_expect):
    """Refuse to proceed on a broken cache. Only asserts on modalities that
    were actually ensured -- e.g. a stale zero 'lidar' array from an earlier
    run doesn't block training if ENSURE_LIDAR=False this run."""
    with h5py.File(path, 'r') as f:
        assert int(f.attrs.get('num_samples', -1)) == n_expect, \
            f"{path.name}: sample-count mismatch"
        results = {}
        if ENSURE_IMAGES or USE_CAMERA:
            assert _images_valid(f, n_expect), f"{path.name}: IMAGES invalid/blank"
            results['img_std'] = float(f['images'][:64].astype(np.float32).std())
        if ENSURE_CSI or USE_CSI:
            assert _csi_valid(f, n_expect), f"{path.name}: CSI invalid/zero"
            results['csi_std'] = float(np.abs(f['csi'][:256]).std())
        if ENSURE_LIDAR or USE_LIDAR:
            assert _lidar_valid(f, n_expect), f"{path.name}: LIDAR invalid/zero"
            results['lidar_std'] = float(f['lidar'][:256].astype(np.float32).std())
    print(f"[OK] Cache verified: {path.name}  (N={n_expect:,})")
    for k, v in results.items():
        print(f"     {k:10s}: {v:.4e}" if 'csi' in k else f"     {k:10s}: {v:.4f}")
    return results


def build_multimodal_cache_if_needed():
    _backfill_legacy_attrs()

    vdata   = _scan_valid_samples()
    samples = vdata['samples']
    N       = len(samples)
    n_cams  = max(NUM_CAMERAS_PER_BS, 1)

    force_csi    = FORCE_FULL_REBUILD or FORCE_REBUILD_CSI
    force_images = FORCE_FULL_REBUILD or FORCE_REBUILD_IMAGES
    force_lidar  = FORCE_FULL_REBUILD or FORCE_REBUILD_LIDAR

    print("=" * 72)
    print(f"CACHE BUILD  ->  {HDF5_FILE.name}")
    print(f"  N={N:,}  ensure(csi={ENSURE_CSI},img={ENSURE_IMAGES},lidar={ENSURE_LIDAR})"
         f"  force(csi={force_csi},img={force_images},lidar={force_lidar})")
    print("=" * 72)

    mode = 'a' if HDF5_FILE.exists() else 'w'
    with h5py.File(HDF5_FILE, mode) as f:

        if ENSURE_CSI:
            if not force_csi and _csi_valid(f, N):
                print(f"[skip ] csi   : already valid ({ANT_TAG})")
            else:
                src = None if force_csi else _find_harvest_source(_csi_valid, N, exclude=HDF5_FILE)
                if src is not None:
                    print(f"[reuse] csi   : harvested from {src.name}")
                    with h5py.File(src, 'r') as fs:
                        _write_full_dataset(f, 'csi', fs['csi'][:],
                                            chunks=(min(500, N), 2, NUM_ANT, NUM_SUB))
                    f.attrs['ant_shape'] = list(ANT_SHAPE)
                    f.attrs['selected_subcarriers'] = list(SELECTED_SUBCARRIERS)
                    f.attrs['subcarrier_mode'] = SUBCARRIER_MODE
                else:
                    print(f"[build] csi   : regenerating via DeepVerse ({ANT_TAG}, slow)")
                    _rebuild_csi(f, samples, N)

        if ENSURE_IMAGES:
            if not force_images and _images_valid(f, N):
                print(f"[skip ] images: already valid ({IMG_TAG})")
            else:
                src = None if force_images else _find_harvest_source(_images_valid, N, exclude=HDF5_FILE)
                if src is not None:
                    print(f"[reuse] images: harvested from {src.name}  "
                         f"(resolution-only match -- antenna shape is irrelevant to images)")
                    with h5py.File(src, 'r') as fs:
                        _write_full_dataset(f, 'images', fs['images'][:],
                                            chunks=(1, n_cams, 3, IMG_H, IMG_W))
                    f.attrs['img_h'] = IMG_H
                    f.attrs['img_w'] = IMG_W
                else:
                    print(f"[build] images: reading PNGs at {IMG_TAG}")
                    _rebuild_images(f, samples, N, n_cams)

        if ENSURE_LIDAR:
            if not force_lidar and _lidar_valid(f, N):
                print(f"[skip ] lidar : already valid")
            else:
                src = None if force_lidar else _find_harvest_source(_lidar_valid, N, exclude=HDF5_FILE)
                if src is not None:
                    print(f"[reuse] lidar : harvested from {src.name}  "
                         f"(antenna- and resolution-independent -- always reusable)")
                    with h5py.File(src, 'r') as fs:
                        _write_full_dataset(f, 'lidar', fs['lidar'][:],
                                            chunks=(min(100, N), LIDAR_POINTS, 3))
                    f.attrs['lidar_points'] = LIDAR_POINTS
                else:
                    print(f"[build] lidar : parsing PCD files (no open3d needed)")
                    _rebuild_lidar(f, samples, N)

        f.attrs['num_samples']  = N
        f.attrs['num_cameras']  = n_cams
        f.attrs['scenario']     = SCENARIO_NAME
        f.attrs['complete']     = True

    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
    _verify_cache(HDF5_FILE, N)
    print(f"[OK] Cache ready: {HDF5_FILE}")


build_multimodal_cache_if_needed()


# ---------------------------------------------------------------------------
# Valid-sample filtering + GPS / offset / CSI-log statistics
# ---------------------------------------------------------------------------
# 1. keep only (scene, bs, ue) triples with real UE/BS positions, so
#    training never sees fabricated (0,0,0) ground truth.
# 2. compute (and cache) GPS stats, BS-relative offset stats and CSI
#    log-domain stats over the clean set only.
# 3. derive one isotropic target scale, OFF_SCALE, from OFF_STD.
#
# OFF_SCALE: normalizing per-axis would weight a low-spread axis (e.g.
# z) far more heavily in the loss than it counts in the reported 3D metric
# -- a 1 m z error could cost ~40x what a 1 m x error does. scaling all axes
# by one scalar (OFF_SCALE = sqrt(mean(OFF_STD^2))) makes the loss term
# exactly the reported mean distance error. OFF_STD is still kept for
# comparability and for the distributional head's bin grid

CLEAN_INDICES_FILE = CACHE_DIR / f"{_SC}_clean_indices.json"
GPS_STATS_FILE     = CACHE_DIR / f"{_SC}_gps_stats_trainonly.json"
OFFSET_STATS_FILE  = CACHE_DIR / f"{_SC}_offset_stats_trainonly.json"
CSI_LOGSTATS_FILE  = CACHE_DIR / f"{_SC}_csi_logstats_{ANT_TAG}_{SUB_TAG}_trainonly.json"

def build_clean_indices():
    """Sorted list of indices into `valid_samples` whose UE-position and
    BS-position entries are both present in the JSON. Cached to disk."""
    if CLEAN_INDICES_FILE.exists():
        with open(CLEAN_INDICES_FILE) as f:
            d = json.load(f)
        print(f"[OK] Loaded clean indices: {len(d['clean']):,} of {d['total']:,} "
              f"(dropped {d['total'] - len(d['clean']):,})")
        return d['clean']

    print("Scanning valid_samples.json for clean entries")
    with open(VALID_SAMPLES_FILE) as f:
        vdata = json.load(f)
    samples      = vdata['samples']
    bs_positions = vdata['bs_positions']
    ue_positions = vdata['ue_positions']

    clean = []
    dropped_no_ue = 0
    dropped_no_bs = 0
    for i, (scene, bs, ue) in enumerate(samples):
        ue_key = f"{scene}_{ue}"
        bs_key = f"{scene}_{bs}"
        if ue_key not in ue_positions:
            dropped_no_ue += 1
            continue
        if bs_key not in bs_positions:
            dropped_no_bs += 1
            continue
        clean.append(i)

    total = len(samples)
    print(f"  Total samples in JSON   : {total:,}")
    print(f"  Dropped (no UE position): {dropped_no_ue:,}")
    print(f"  Dropped (no BS position): {dropped_no_bs:,}")
    print(f"  Clean samples retained  : {len(clean):,}")
    if len(clean) < 100:
        raise RuntimeError(
            f"Only {len(clean)} clean samples remain. Something is wrong with "
            f"the preprocessing -- check {VALID_SAMPLES_FILE}."
        )

    with open(CLEAN_INDICES_FILE, 'w') as f:
        json.dump({
            'clean': clean,
            'total': total,
            'dropped_no_ue': dropped_no_ue,
            'dropped_no_bs': dropped_no_bs,
        }, f)
    print(f"[OK] Saved clean indices -> {CLEAN_INDICES_FILE}")
    return clean


def build_scene_block_split(clean_idx, valid_samples_json, train_ratio, val_ratio):
    """Scene-contiguous split (replaces a random per-sample permutation).

    For a single-BS/single-UE scenario like DT31, clean_idx indexes one
    continuous vehicle trajectory where consecutive scenes are ~0.5 m apart.
    A random shuffle interleaves train/val/test almost frame-by-frame --
    verified empirically on DT31, >90% of test samples had a training
    sample one scene-step away (mean nearest-neighbour distance ~0.08 m) --
    so the model could memorize/interpolate instead of learning to localize,
    which is why earlier splits beat the paper's baseline by 10-100x.

    This orders samples by true (scene, bs, ue) capture order and cuts three
    contiguous blocks, so val/test are never adjacent in time to training.

    Caveat: DT31's route is a repeated back-and-forth shuttle over the same
    ~81 m of street (56 passes), so this removes the frame-adjacency leak but
    test scenes still cover locations driven over in earlier passes -- it
    answers "does this generalize to a later time on this fixed RSU's
    street", matching how such a system is actually deployed and the
    protocol implied by the paper's 70/30 split. A stricter spatial hold-out
    across all passes would be a separate, harder diagnostic.
    """
    with open(valid_samples_json) as f:
        vdata = json.load(f)
    samples = vdata['samples']   # each entry: [scene, bs, ue]
    ordered = sorted(clean_idx, key=lambda i: tuple(samples[i]))

    n_clean = len(ordered)
    n_train = int(n_clean * train_ratio)
    n_val   = int(n_clean * val_ratio)

    train_idx = ordered[:n_train]
    val_idx   = ordered[n_train:n_train + n_val]
    test_idx  = ordered[n_train + n_val:]

    print(f"[OK] Scene-block split (Option A): "
          f"Train {len(train_idx):,} (scenes 0-{samples[train_idx[-1]][0]}) | "
          f"Val {len(val_idx):,} (scenes {samples[val_idx[0]][0]}-{samples[val_idx[-1]][0]}) | "
          f"Test {len(test_idx):,} (scenes {samples[test_idx[0]][0]}-{samples[test_idx[-1]][0]})")
    return train_idx, val_idx, test_idx


def compute_or_load_gps_stats(clean_idx):
    """Mean/std of UE positions over the clean sample set only. Cached so
    every run uses the same stats regardless of split."""
    if GPS_STATS_FILE.exists():
        with open(GPS_STATS_FILE) as f:
            s = json.load(f)
        print(f"[OK] Loaded GPS normalization stats from {GPS_STATS_FILE}")
        return np.asarray(s['mean'], dtype=np.float32), np.asarray(s['std'], dtype=np.float32)

    print("Computing GPS normalization stats over clean samples")
    with open(VALID_SAMPLES_FILE) as f:
        vdata = json.load(f)
    samples      = vdata['samples']
    ue_positions = vdata['ue_positions']

    coords = np.zeros((len(clean_idx), 3), dtype=np.float64)
    for j, i in enumerate(clean_idx):
        scene, bs, ue = samples[i]
        coords[j] = ue_positions[f"{scene}_{ue}"]   # guaranteed to exist by filter

    mean = coords.mean(axis=0)
    std  = coords.std(axis=0)
    std[std < 1e-6] = 1.0      # protect against degenerate axes (e.g. flat z)

    stats = {
        'mean': mean.tolist(),
        'std':  std.tolist(),
        'min':  coords.min(axis=0).tolist(),
        'max':  coords.max(axis=0).tolist(),
        'num_clean_samples': int(len(clean_idx)),
    }
    with open(GPS_STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"[OK] Saved GPS stats -> {GPS_STATS_FILE}")
    print(f"     mean={mean.round(2).tolist()}  std={std.round(2).tolist()}")
    return mean.astype(np.float32), std.astype(np.float32)


def compute_or_load_offset_stats(clean_idx):
    """Mean/std of the BS-relative offset Delta = UE - BS over the clean
    set. This is the regression target, since CSI measured at a BS is a
    function of relative geometry only. Cached to disk."""
    if OFFSET_STATS_FILE.exists():
        with open(OFFSET_STATS_FILE) as f:
            s = json.load(f)
        print(f"[OK] Loaded OFFSET stats from {OFFSET_STATS_FILE}")
        return np.asarray(s['mean'], dtype=np.float32), np.asarray(s['std'], dtype=np.float32)

    print("Computing OFFSET (UE - BS) normalization stats over clean samples")
    with open(VALID_SAMPLES_FILE) as f:
        vdata = json.load(f)
    samples      = vdata['samples']
    ue_positions = vdata['ue_positions']
    bs_positions = vdata['bs_positions']

    offs = np.zeros((len(clean_idx), 3), dtype=np.float64)
    for j, i in enumerate(clean_idx):
        scene, bs, ue = samples[i]
        ue_p = np.asarray(ue_positions[f"{scene}_{ue}"], dtype=np.float64)
        bs_p = np.asarray(bs_positions[f"{scene}_{bs}"], dtype=np.float64)
        offs[j] = ue_p - bs_p

    mean = offs.mean(axis=0)
    std  = offs.std(axis=0)
    std[std < 1e-6] = 1.0

    stats = {
        'mean': mean.tolist(),
        'std':  std.tolist(),
        'min':  offs.min(axis=0).tolist(),
        'max':  offs.max(axis=0).tolist(),
        'num_clean_samples': int(len(clean_idx)),
    }
    with open(OFFSET_STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"[OK] Saved OFFSET stats -> {OFFSET_STATS_FILE}")
    print(f"     mean={mean.round(2).tolist()}  std={std.round(2).tolist()}")
    return mean.astype(np.float32), std.astype(np.float32)


def compute_or_load_csi_log_stats(clean_idx, max_samples=50000):
    """Mean/std of (a) log10(|h|/CSI_STD + 1e-6) over all (ant, sub)
    and (b) log10(|FFT2(h/CSI_STD)| + 1e-6) over all angle-delay bins,
    estimated on a random subsample of the clean set. Cached so every run --
    train or eval-only -- standardizes the log-domain channels identically.

    log is used because |h| spans ~150 dB; log10|h| is approximately affine
    in log-distance (path loss), so it's the right coordinate for a distance
    regressor and has near-Gaussian spread, unlike linear |h|."""
    if CSI_LOGSTATS_FILE.exists():
        with open(CSI_LOGSTATS_FILE) as f:
            s = json.load(f)
        print(f"[OK] Loaded CSI log-domain stats from {CSI_LOGSTATS_FILE}")
        return (float(s['logmag_mean']), float(s['logmag_std']),
                float(s['fftlog_mean']), float(s['fftlog_std']))

    print(f"Computing CSI log-domain stats (subsample <= {max_samples:,})")
    rng = np.random.RandomState(SEED)
    idx = np.asarray(clean_idx, dtype=np.int64)
    if len(idx) > max_samples:
        idx = rng.choice(idx, size=max_samples, replace=False)
    idx = np.sort(idx)

    n   = len(idx)
    s1 = s2 = 0.0   # running sums for log|h|
    f1 = f2 = 0.0   # running sums for log|FFT2(h)|
    cnt = 0
    t0 = time.time()
    with h5py.File(HDF5_FILE, 'r') as f:
        dset = f['csi']
        CH = 4096                                  
        for a in range(0, n, CH):
            block = dset[idx[a:a + CH]]            # (B, 2, N_ant, N_sub)
            H = (block[:, 0] + 1j * block[:, 1]) / CSI_STD
            mag = np.abs(H)
            lm  = np.log10(mag + 1e-6)
            A   = np.fft.fft2(H, axes=(-2, -1)) / np.sqrt(H.shape[-1] * H.shape[-2])
            fl  = np.log10(np.abs(A) + 1e-6)
            s1 += lm.sum();  s2 += (lm ** 2).sum()
            f1 += fl.sum();  f2 += (fl ** 2).sum()
            cnt += lm.size
    lm_mean = s1 / cnt
    lm_std  = max(np.sqrt(max(s2 / cnt - lm_mean ** 2, 0.0)), 1e-6)
    fl_mean = f1 / cnt
    fl_std  = max(np.sqrt(max(f2 / cnt - fl_mean ** 2, 0.0)), 1e-6)
    print(f"  log10|h|     : mean={lm_mean:.4f}  std={lm_std:.4f}")
    print(f"  log10|FFT2 h|: mean={fl_mean:.4f}  std={fl_std:.4f}")
    print(f"  elapsed: {time.time() - t0:.1f}s over {n:,} samples")

    with open(CSI_LOGSTATS_FILE, 'w') as f:
        json.dump({
            'logmag_mean': float(lm_mean), 'logmag_std': float(lm_std),
            'fftlog_mean': float(fl_mean), 'fftlog_std': float(fl_std),
            'csi_std_used': float(CSI_STD),
            'num_samples_used': int(n),
        }, f, indent=2)
    print(f"[OK] Saved CSI log stats -> {CSI_LOGSTATS_FILE}")
    return float(lm_mean), float(lm_std), float(fl_mean), float(fl_std)


CLEAN_INDICES = build_clean_indices()

TRAIN_IDX, VAL_IDX, TEST_IDX = build_scene_block_split(
    CLEAN_INDICES, VALID_SAMPLES_FILE, TRAIN_RATIO, VAL_RATIO)

GPS_MEAN, GPS_STD = compute_or_load_gps_stats(TRAIN_IDX)
with open(GPS_STATS_FILE) as f:
    _s = json.load(f)
GPS_MIN = np.asarray(_s['min'], dtype=np.float32)
GPS_MAX = np.asarray(_s['max'], dtype=np.float32)

OFF_MEAN, OFF_STD = compute_or_load_offset_stats(TRAIN_IDX)
LOGMAG_MEAN, LOGMAG_STD, FFTLOG_MEAN, FFTLOG_STD = \
    compute_or_load_csi_log_stats(TRAIN_IDX)

OFF_SCALE = float(np.sqrt(np.mean(np.asarray(OFF_STD, dtype=np.float64) ** 2)))
if not np.isfinite(OFF_SCALE) or OFF_SCALE < 1e-6:
    OFF_SCALE = 1.0
    print("[!]  Degenerate OFF_STD -- falling back to OFF_SCALE = 1.0 m")

AXIS_SCALE = np.maximum(
    np.asarray(OFF_STD, dtype=np.float64) / OFF_SCALE, 0.05).astype(np.float32)

print(f"  GPS_MEAN  = {GPS_MEAN.tolist()}")
print(f"  GPS_STD   = {GPS_STD.tolist()}")
print(f"  GPS_MIN   = {GPS_MIN.tolist()}")
print(f"  GPS_MAX   = {GPS_MAX.tolist()}")
print(f"  OFF_MEAN  = {OFF_MEAN.tolist()}")
print(f"  OFF_STD   = {OFF_STD.tolist()}      (per-axis; reporting only)")
print(f"  OFF_SCALE = {OFF_SCALE:.4f} m       (v5 ISOTROPIC target scale)")
print(f"  AXIS_SCALE= {AXIS_SCALE.round(4).tolist()}  (bin-grid width per axis)")
print(f"  1 normalized unit = {OFF_SCALE:.3f} m  ->  a 1.0 m error is "
      f"{1.0 / OFF_SCALE:.5f} in loss units")
print(f"  Clean samples: {len(CLEAN_INDICES):,}")


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------
# HDF5 cache format is untouched (raw Re/Im CSI, images, LiDAR); everything
# here is computed on the fly, so an older cache still work.
#
# CSI features: 12 channels (ULA) / 14 (UPA). beyond raw Re/Im + log-mag:
#   ch 6-7   cos/sin of angle(A), A = FFT2(h) -- beamspace phase, gives the
#            sub-bin AoA/delay offset that log|A| alone quantizes away.
#   ch 8-9   cos/sin of angle(h[a,k+1] conj(h[a,k])) -- subcarrier-diff
#            phase, proportional to delay/range.
#   ch 10-11 (ULA) cos/sin of angle(h[a+1,k] conj(h[a,k])) -- antenna-diff
#            phase, proportional to sin(AoA).
#   both differentials are invariant to the global carrier phase, so
#   they're nuisance-free and need no new normalization stats.
#
# UPA fix: for a 2D (rows, cols) array, a flat 1D diff mixes the last
# antenna of one row with the first of the next. _csi_features() reshapes
# to the true grid instead and computes two real nearest-neighbour diffs:
#   ch 10-11 column-direction diff (azimuth cue)
#   ch 12-13 row-direction diff (elevation cue)
# CSI_FEAT_CH is 14 for UPA / 12 for ULA; the ULA branch is unchanged, so
# switching back to (16,1) reproduces the original numbers.
#
# target = (UE - BS - OFF_MEAN) / OFF_SCALE, a scalar scale, so
# Euclidean distance in target space is metres on every axis equally. also
# keeps BS-relative target, train-time phase rotation and complex AWGN aug.

_IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
_IMNET_STD  = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)

class MultimodalGPSDataset(Dataset):
    """
    Inputs (features dict):
      - 'csi_feat'   : (12, N_ant, N_sub)            [if USE_CSI]  -- see header
      - 'bs_pos'     : (3,)  normalized BS coordinates  [if USE_BS_POS]
      - 'bs_pos_raw' : (3,)  raw BS coordinates (always present -- needed to
                       reconstruct UE = BS + offset at evaluation time)
      - 'images'     : (N_cams, 3, H, W)             [if USE_CAMERA]
      - 'lidar'      : (LIDAR_POINTS, 3)             [if USE_LIDAR]

    Targets returned by __getitem__:
      - target  : (3,)  isotropically-scaled offset (UE - BS) -- the
                  regression target; 1 unit == OFF_SCALE metres on every axis
      - features: input dict (as above)
      - ue_raw  : (3,)  raw UE coordinates in metres (for metrics)

    The constructor requires every index in `indices` to have both a valid
    UE-position and BS-position entry in the JSON
    clean-index filter; re-asserted here as defence-in-depth so the dataset
    never silently fabricates a (0,0,0) ground truth.
    """

    def __init__(self, hdf5_path, valid_samples_json, indices,
                 img_size=64, lidar_points=1024,
                 num_cameras_per_bs=4,
                 use_csi=True, use_bs_pos=True,
                 use_camera=False, use_lidar=False,
                 use_all_cameras=False,
                 csi_std=4.505086e-07,
                 logmag_mean=0.0, logmag_std=1.0,
                 fftlog_mean=0.0, fftlog_std=1.0,
                 off_mean=None, off_scale=1.0,
                 gps_mean=None, gps_std=None,
                 bs_pos_norm_scale=350.0,
                 augment=False,
                 aug_phase=True, aug_noise_p=0.5, aug_snr_db=(15.0, 35.0),
                 is_upa=None, ant_rows=None, ant_cols=None):
        self.hdf5_path          = str(hdf5_path)
        self.indices            = indices
        self.img_size           = img_size
        self.lidar_points       = lidar_points
        self.num_cameras_per_bs = num_cameras_per_bs
        self.use_csi            = use_csi
        self.use_bs_pos         = use_bs_pos
        self.use_camera         = use_camera
        self.use_lidar          = use_lidar
        self.use_all_cameras    = use_all_cameras
        self.csi_std            = csi_std
        self.logmag_mean        = logmag_mean
        self.logmag_std         = logmag_std
        self.fftlog_mean        = fftlog_mean
        self.fftlog_std         = fftlog_std
        self.bs_pos_norm_scale  = bs_pos_norm_scale
        self.off_mean           = np.asarray(off_mean, dtype=np.float32)
        self.off_scale          = float(off_scale)
        self.gps_mean           = gps_mean
        self.gps_std            = gps_std
        self.augment            = augment
        self.aug_phase          = aug_phase
        self.aug_noise_p        = aug_noise_p
        self.aug_snr_db         = aug_snr_db
        self.h5file             = None
        
        self.is_upa  = IS_UPA  if is_upa  is None else is_upa
        self.ant_rows = ANT_ROWS if ant_rows is None else ant_rows
        self.ant_cols = ANT_COLS if ant_cols is None else ant_cols
        if self.is_upa:
            assert self.ant_rows * self.ant_cols == NUM_ANT, \
                f"ant_rows*ant_cols ({self.ant_rows}*{self.ant_cols}) must equal NUM_ANT ({NUM_ANT})"

        with open(valid_samples_json) as f:
            vdata = json.load(f)
        self.valid_samples = vdata['samples']
        self.bs_positions  = vdata['bs_positions']
        self.ue_positions  = vdata['ue_positions']

        for i in indices:
            s, b, u = self.valid_samples[i]
            if f"{s}_{u}" not in self.ue_positions:
                raise KeyError(
                    f"Sample index {i} -> (scene={s}, bs={b}, ue={u}) is missing "
                    f"a UE position in {valid_samples_json}. Did you forget to "
                    f"pass clean indices"
                )
            if f"{s}_{b}" not in self.bs_positions:
                raise KeyError(
                    f"Sample index {i} -> (scene={s}, bs={b}, ue={u}) is missing "
                    f"a BS position in {valid_samples_json}."
                )

    def __len__(self):
        return len(self.indices)

    def _open_h5(self):
        if self.h5file is None:
            self.h5file = h5py.File(self.hdf5_path, 'r')

    def _csi_features(self, csi_raw):
        """Raw cached (2, N_ant, N_sub) Re/Im -> (CSI_FEAT_CH, N_ant, N_sub)
        engineered features. 12 channels for a ULA, 14 for a UPA -- only the
        antenna-differential-phase channels depend on geometry, everything
        else is identical. All transforms are pointwise, small local
        differences, or an O(N log N) FFT over a small grid, negligible
        next to the HDF5 image read."""
        H = (csi_raw[0] + 1j * csi_raw[1]).astype(np.complex64)

        if self.augment:
           
            if self.aug_phase:
                theta = np.random.uniform(0.0, 2.0 * np.pi)
                H = H * np.exp(1j * theta).astype(np.complex64)
            if self.aug_noise_p > 0 and np.random.rand() < self.aug_noise_p:
                p_sig = float(np.mean(np.abs(H) ** 2))
                if p_sig > 0:
                    snr_db  = np.random.uniform(*self.aug_snr_db)
                    p_noise = p_sig / (10.0 ** (snr_db / 10.0))
                    sigma   = np.sqrt(p_noise / 2.0)
                    H = H + (sigma * (np.random.randn(*H.shape)
                                      + 1j * np.random.randn(*H.shape))
                             ).astype(np.complex64)

        Hs  = H / self.csi_std                       # unit-scale linear field
        mag = np.abs(Hs)

        ri      = np.stack([Hs.real, Hs.imag], axis=0)                  # ch 0-1
        logmag  = (np.log10(mag + 1e-6) - self.logmag_mean) / self.logmag_std   # ch 2
        ph      = np.angle(Hs)
        cos_ph  = np.cos(ph)                                            # ch 3
        sin_ph  = np.sin(ph)                                            # ch 4
        A       = np.fft.fft2(Hs) / np.sqrt(Hs.shape[0] * Hs.shape[1])
        fft_lm  = (np.log10(np.abs(A) + 1e-6) - self.fftlog_mean) / self.fftlog_std  # ch 5

        a_ph    = np.angle(A)
        cos_aph = np.cos(a_ph)                                          # ch 6
        sin_aph = np.sin(a_ph)                                          # ch 7

        d_sub = Hs[:, 1:] * np.conj(Hs[:, :-1])
        d_sub = np.concatenate([d_sub, d_sub[:, -1:]], axis=1)
        p_sub = np.angle(d_sub)
        cos_ds = np.cos(p_sub)                                          # ch 8
        sin_ds = np.sin(p_sub)                                          # ch 9

        if not self.is_upa:
            d_ant = Hs[1:, :] * np.conj(Hs[:-1, :])
            d_ant = np.concatenate([d_ant, d_ant[-1:, :]], axis=0)
            p_ant = np.angle(d_ant)
            cos_da = np.cos(p_ant)                                      # ch 10
            sin_da = np.sin(p_ant)                                      # ch 11

            feat = np.stack([
                ri[0],   ri[1],   logmag,  cos_ph,  sin_ph,  fft_lm,
                cos_aph, sin_aph, cos_ds,  sin_ds,  cos_da,  sin_da,
            ], axis=0).astype(np.float32)                  # (12, N_ant, N_sub)
        else:
            R, C = self.ant_rows, self.ant_cols
            Hg = Hs.reshape(R, C, Hs.shape[1])                          # (R, C, N_sub)

            d_col = Hg[:, 1:, :] * np.conj(Hg[:, :-1, :])                # (R, C-1, S)
            d_col = np.concatenate([d_col, d_col[:, -1:, :]], axis=1)    # edge-replicate -> (R, C, S)
            p_col = np.angle(d_col).reshape(Hs.shape)                   # back to (N_ant, N_sub)
            cos_dx = np.cos(p_col)                                      # ch 10 (azimuth cue)
            sin_dx = np.sin(p_col)                                      # ch 11

            d_row = Hg[1:, :, :] * np.conj(Hg[:-1, :, :])                # (R-1, C, S)
            d_row = np.concatenate([d_row, d_row[-1:, :, :]], axis=0)    # edge-replicate -> (R, C, S)
            p_row = np.angle(d_row).reshape(Hs.shape)                   # back to (N_ant, N_sub)
            cos_dy = np.cos(p_row)                                      # ch 12 (elevation cue)
            sin_dy = np.sin(p_row)                                      # ch 13

            feat = np.stack([
                ri[0],   ri[1],   logmag,  cos_ph,  sin_ph,  fft_lm,
                cos_aph, sin_aph, cos_ds,  sin_ds,
                cos_dx,  sin_dx,  cos_dy,  sin_dy,
            ], axis=0).astype(np.float32)                  # (14, N_ant, N_sub)

        return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    def __getitem__(self, idx):
        self._open_h5()
        i = self.indices[idx]
        scene, bs, ue = self.valid_samples[i]

        ue_raw     = np.asarray(self.ue_positions[f"{scene}_{ue}"], dtype=np.float32)
        bs_pos_raw = np.asarray(self.bs_positions[f"{scene}_{bs}"], dtype=np.float32)

        off_norm = ((ue_raw - bs_pos_raw) - self.off_mean) / self.off_scale

        features = {}

        if self.use_csi:
            csi = self.h5file['csi'][i]                       # (2, N_ant, N_sub) raw
            features['csi_feat'] = torch.from_numpy(self._csi_features(csi))

        # BS position
        features['bs_pos_raw'] = torch.from_numpy(bs_pos_raw)   # always (for denorm)
        if self.use_bs_pos:
            features['bs_pos'] = torch.from_numpy(bs_pos_raw / self.bs_pos_norm_scale)

        # camera
        if self.use_camera:
            images = self.h5file['images'][i]              # uint8
            if not self.use_all_cameras:
                images = images[0:1]
            images = images.astype(np.float32) / 255.0
            images = (images - _IMNET_MEAN) / _IMNET_STD
            features['images'] = torch.from_numpy(images)

        # lidar
        if self.use_lidar:
            features['lidar'] = torch.from_numpy(
                self.h5file['lidar'][i].astype(np.float32)
            )

        target   = torch.from_numpy(off_norm.astype(np.float32))  # scaled offset
        ue_raw_t = torch.from_numpy(ue_raw)                        # raw, for metrics
        return target, features, ue_raw_t


def collate_fn(batch):
    targets = torch.stack([b[0] for b in batch])
    ue_raw  = torch.stack([b[2] for b in batch])
    keys = batch[0][1].keys()
    features = {k: torch.stack([b[1][k] for b in batch]) for k in keys}
    return targets, features, ue_raw


def _verify_csi_feature_invariances(tol=1e-4):
    rng = np.random.RandomState(0)
    raw = rng.randn(2, NUM_ANT, NUM_SUB).astype(np.float32) * CSI_STD
    probe = MultimodalGPSDataset.__new__(MultimodalGPSDataset)
    probe.augment      = False
    probe.csi_std      = CSI_STD
    probe.logmag_mean, probe.logmag_std = LOGMAG_MEAN, LOGMAG_STD
    probe.fftlog_mean, probe.fftlog_std = FFTLOG_MEAN, FFTLOG_STD
    probe.is_upa   = IS_UPA
    probe.ant_rows = ANT_ROWS
    probe.ant_cols = ANT_COLS

    f0 = probe._csi_features(raw)
    theta = 0.7231
    Hr = ((raw[0] + 1j * raw[1]) * np.exp(1j * theta)).astype(np.complex64)
    f1 = probe._csi_features(np.stack([Hr.real, Hr.imag], axis=0).astype(np.float32))

    invariant_ch = (2, 5, 8, 9, 10, 11) + ((12, 13) if IS_UPA else ())

    c, s = math.cos(theta), math.sin(theta)
    ok = True
    for (a, b) in ((0, 1), (3, 4), (6, 7)):          # pairs that rotate
        exp_a = f0[a] * c - f0[b] * s
        exp_b = f0[a] * s + f0[b] * c
        e = max(np.abs(f1[a] - exp_a).max(), np.abs(f1[b] - exp_b).max())
        ok &= bool(e < tol)
        print(f"  ch({a},{b}) rotate by theta      : max err {e:.2e}"
              f"  {'OK' if e < tol else 'FAIL'}")
    for ch in invariant_ch:                          # channels that are invariant
        e = float(np.abs(f1[ch] - f0[ch]).max())
        ok &= bool(e < tol)
        print(f"  ch{ch:<2d} invariant to theta      : max err {e:.2e}"
              f"  {'OK' if e < tol else 'FAIL'}")
    assert f0.shape[0] == CSI_FEAT_CH, \
        f"_csi_features produced {f0.shape[0]} channels, expected CSI_FEAT_CH={CSI_FEAT_CH}"
    if not ok:
        raise RuntimeError("CSI feature invariance self-check FAILED -- "
                           "phase TTA would be incorrect.")
    return ok


if USE_CSI:
    print("[..] Verifying CSI feature phase-invariance properties")
    _verify_csi_feature_invariances()

print(f"[OK] MultimodalGPSDataset defined "
      f"(v6: {CSI_FEAT_CH}-ch CSI features [{'UPA' if IS_UPA else 'ULA'}], "
      f"isotropic offset target, aug)")

def diagnose_antenna_ordering(hdf5_path=None, num_samples=500, seed=0):
    """Heuristic sanity check for ANT_ORDER on a non-square UPA.

    Reshapes the raw antenna axis into the (rows, cols) grid under each
    candidate ordering, takes a true 2D spatial FFT per subcarrier, and
    measures how peaked the resulting angular spectrum is (max/mean power,
    averaged over subcarriers and sampled scenes). A physically coherent
    array, reshaped with the ordering matching its real layout, concentrates
    energy into a narrower angular beam than a reshape that scrambles true
    neighbours into false ones -- the ordering with the higher average
    peak/mean ratio is the suggested match.

    This is a heuristic, not a proof; for certainty check how
    dv.comm.bs_antenna enumerates elements relative to comm.coeffs in the
    installed DeepVerse package. No-op when IS_UPA is False or ANT_SHAPE is
    square (both orderings are then mathematically identical)."""
    if not IS_UPA:
        print("[diagnose_antenna_ordering] ANT_SHAPE is a ULA -- ordering is irrelevant.")
        return None
    if ANT_SHAPE[0] == ANT_SHAPE[1]:
        print(f"[diagnose_antenna_ordering] ANT_SHAPE={ANT_SHAPE} is square -- "
              f"'dim0_outer' and 'dim1_outer' are mathematically identical, no check needed.")
        return None

    path = Path(hdf5_path) if hdf5_path is not None else HDF5_FILE
    if not path.exists():
        print(f"[diagnose_antenna_ordering] cache not found at {path} -- build it first.")
        return None

    rng = np.random.RandomState(seed)
    with h5py.File(path, 'r') as f:
        if 'csi' not in f:
            print("[diagnose_antenna_ordering] no 'csi' dataset in cache.")
            return None
        N = f['csi'].shape[0]
        n = min(num_samples, N)
        sel = np.sort(rng.choice(N, n, replace=False))
        raw = f['csi'][sel]                                   # (n, 2, NUM_ANT, NUM_SUB)

    H = (raw[:, 0] + 1j * raw[:, 1]).astype(np.complex64)      # (n, NUM_ANT, NUM_SUB)

    def _peak_to_mean(order):
        r, c = ANT_SHAPE if order == 'dim0_outer' else (ANT_SHAPE[1], ANT_SHAPE[0])
        Hg = H.reshape(n, r, c, NUM_SUB)
        A  = np.fft.fft2(Hg, axes=(1, 2))                      # (n, r, c, NUM_SUB)
        P  = np.abs(A) ** 2
        ratio = P.max(axis=(1, 2)) / np.clip(P.mean(axis=(1, 2)), 1e-20, None)
        return float(ratio.mean())

    r0 = _peak_to_mean('dim0_outer')
    r1 = _peak_to_mean('dim1_outer')
    best = 'dim0_outer' if r0 >= r1 else 'dim1_outer'
    print(f"[diagnose_antenna_ordering] n={n} samples from {path.name}")
    print(f"  peak/mean power, dim0_outer : {r0:.3f}")
    print(f"  peak/mean power, dim1_outer : {r1:.3f}")
    tag = "  (matches current ANT_ORDER)" if best == ANT_ORDER else \
          "  <-- DIFFERS from current ANT_ORDER; consider switching"
    print(f"  suggested ANT_ORDER          : '{best}'{tag}")
    print("  (heuristic peakiness check -- confirm against DeepVerse's antenna "
          "indexing source if you need certainty)")
    return best


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------
#
#   - CSIEncoder   : (12, N_ant, N_sub) -> dense via CoordConv + stride-1
#                    residual blocks (no pooling) -> Flatten -> Linear x2,
#                    a full-rank map of every (antenna, subcarrier).
#                    tokens via 1x1 proj of the same trunk + learned
#                    positional embedding per bin.
#   - BSPosEncoder : Fourier features of normalized BS xyz.
#   - CameraEncoder: CNN backbone shared between pooled dense vector and
#                    spatial tokens (+ camera-ID embedding).
#   - LidarEncoder : PointNet features -> dense (attn-pool + max-pool) and
#                    tokens (PMA/Set-Transformer, M learnable seeds + mask).
#                    cache already stores roughly unit-scale points, so a
#                    LayerNorm after the first projection is enough.
class ResBlock2d(nn.Module):
    """3x3 -> 3x3 residual block, stride 1 (the CSI grid is only 16x8 --
    never downsampled, only mixed)."""
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(ch)
        self.act   = nn.GELU()

    def forward(self, x):
        h = self.act(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.act(x + h)


class CSIEncoder(nn.Module):
    """(B, C_in, N_ant, N_sub) -> (dense (B, embed_dim), tokens (B, A*S, token_dim)).

    CoordConv: extra channels carrying the normalized antenna/subcarrier
    position, since angle-delay information lives in the absolute position
    of spectral peaks and translation-invariant convolutions are blind to it.

    For a ULA a single antenna-index coordinate (linspace(-1,1,N_ant)) is a
    faithful 1D position. For a UPA the flat N_ant axis is a row-major
    flatten of a (ant_rows, ant_cols) grid, so one such coordinate isn't
    spatially meaningful (index 3 and 4 aren't adjacent in space), so
    is_upa=True emits two coordinate channels (row, col) instead of one.
    """
    def __init__(self, embed_dim=384, token_dim=256, num_ant=16, num_sub=8,
                 in_ch=12, width=96, num_blocks=4, hidden=768, dropout=0.05,
                 flatten_ch=None, is_upa=None, ant_rows=None, ant_cols=None):
        super().__init__()
        self.num_ant, self.num_sub = num_ant, num_sub
        self.is_upa = IS_UPA if is_upa is None else is_upa
        if flatten_ch is None:
            flatten_ch = width

        if self.is_upa:
            rows = ANT_ROWS if ant_rows is None else ant_rows
            cols = ANT_COLS if ant_cols is None else ant_cols
            assert rows * cols == num_ant, \
                f"ant_rows*ant_cols ({rows}*{cols}) must equal num_ant ({num_ant})"
            idx = torch.arange(num_ant)
            row_c = torch.linspace(-1., 1., rows)[idx // cols]            # (num_ant,)
            col_c = torch.linspace(-1., 1., cols)[idx % cols]             # (num_ant,)
            ant_row = row_c.view(1, 1, num_ant, 1).expand(1, 1, num_ant, num_sub)
            ant_col = col_c.view(1, 1, num_ant, 1).expand(1, 1, num_ant, num_sub)
            sub = torch.linspace(-1., 1., num_sub).view(1, 1, 1, num_sub).expand(1, 1, num_ant, num_sub)
            self.register_buffer('coord', torch.cat([ant_row, ant_col, sub], dim=1))  # (1,3,A,S)
            coord_ch = 3
        else:
            ant = torch.linspace(-1., 1., num_ant).view(1, 1, num_ant, 1).expand(1, 1, num_ant, num_sub)
            sub = torch.linspace(-1., 1., num_sub).view(1, 1, 1, num_sub).expand(1, 1, num_ant, num_sub)
            self.register_buffer('coord', torch.cat([ant, sub], dim=1))          # (1,2,A,S)
            coord_ch = 2

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch + coord_ch, width, 3, padding=1),
            nn.BatchNorm2d(width), nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResBlock2d(width) for _ in range(num_blocks)])

        self.bottleneck = nn.Conv2d(width, flatten_ch, 1)
        self.dense = nn.Sequential(
            nn.Flatten(),                                    # flatten_ch * A * S
            nn.Linear(flatten_ch * num_ant * num_sub, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
        )
        self.proj    = nn.Conv2d(width, token_dim, 1)
        self.pos_emb = nn.Parameter(torch.randn(1, num_ant * num_sub, token_dim) * 0.02)

    def forward(self, csi_feat):
        B = csi_feat.size(0)
        x = torch.cat([csi_feat, self.coord.expand(B, -1, -1, -1)], dim=1)
        x = self.blocks(self.stem(x))                                        # (B,W,A,S)
        emb = self.dense(self.bottleneck(x))                                 # (B,embed_dim)
        tok = self.proj(x).flatten(2).permute(0, 2, 1) + self.pos_emb        # (B,A*S,d)
        return emb, tok


class BSPosEncoder(nn.Module):
    """Fourier-feature (NeRF-style) encoding of the normalized BS xyz ->
    (dense (B, embed_dim), token (B, 1, token_dim))."""
    def __init__(self, embed_dim=96, token_dim=256, n_freq=8):
        super().__init__()
        self.register_buffer('freqs', (2.0 ** torch.arange(n_freq)) * math.pi)
        in_dim = 3 + 3 * 2 * n_freq
        self.dense = nn.Sequential(
            nn.Linear(in_dim, 128), nn.GELU(),
            nn.Linear(128, embed_dim),
        )
        self.tok = nn.Sequential(
            nn.Linear(in_dim, 128), nn.GELU(),
            nn.Linear(128, token_dim),
        )

    def _fourier(self, bs_pos):
        x = bs_pos.unsqueeze(-1) * self.freqs.to(bs_pos.dtype)               # (B,3,F)
        return torch.cat([bs_pos, torch.sin(x).flatten(1), torch.cos(x).flatten(1)], dim=1)

    def forward(self, bs_pos):
        ff = self._fourier(bs_pos)
        return self.dense(ff), self.tok(ff).unsqueeze(1)

class CameraEncoder(nn.Module):
    def __init__(self, token_dim=256, img_hw=(540,960), dense_dim=256, reduce_ch=None):
        super().__init__()
        if reduce_ch is None:
            reduce_ch = 32          # disables the bottleneck: original width, as before
        self.backbone = nn.Sequential(
            nn.Conv2d(3,   32, 3, stride=2, padding=1), nn.BatchNorm2d(32),  nn.GELU(),
            nn.Conv2d(32,  64, 3, stride=2, padding=1), nn.BatchNorm2d(64),  nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128,128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128,128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
        )
        with torch.no_grad():
            fh, fw = self.backbone(torch.zeros(1,3,*img_hw)).shape[-2:]
        self.fh, self.fw = int(fh), int(fw)
        print(f"  CameraEncoder: {img_hw} -> {self.fh}x{self.fw} = {self.fh*self.fw} tokens")
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.reduce  = nn.Conv2d(128, reduce_ch, 1)
        self.spatial = nn.Linear(reduce_ch*self.fh*self.fw, dense_dim)
        self.fc      = nn.Linear(128+dense_dim, dense_dim)
        self.proj    = nn.Conv2d(128, token_dim, 1)
        self.pos_emb = nn.Parameter(torch.randn(1, self.fh*self.fw, token_dim)*0.02)


    def forward(self, x):
        f    = self.backbone(x)
        gap  = self.pool(f).flatten(1)
        spat = self.spatial(self.reduce(f).flatten(1))
        emb  = self.fc(torch.cat([gap, spat], dim=1))
        tok  = self.proj(f).flatten(2).permute(0,2,1) + self.pos_emb
        return emb, tok

class MultiCameraEncoder(nn.Module):
    def __init__(self, num_cameras=4, embed_dim=192, token_dim=256, img_hw=(540,960),
                 reduce_ch=None):
        super().__init__()
        self.num_cameras = num_cameras
        self.encoders = nn.ModuleList([CameraEncoder(token_dim, img_hw, reduce_ch=reduce_ch)
                                       for _ in range(num_cameras)])
        self.fusion = nn.Linear(256*num_cameras, embed_dim)
        self.cam_id = nn.Parameter(torch.randn(num_cameras, 1, token_dim)*0.02)


    def forward(self, images):
        embs, toks = [], []
        for i, enc in enumerate(self.encoders):
            e, t = enc(images[:, i]); embs.append(e); toks.append(t + self.cam_id[i].to(t.dtype))
        return self.fusion(torch.cat(embs, dim=1)), torch.cat(toks, dim=1)

class SingleCameraEncoder(nn.Module):
    def __init__(self, embed_dim=192, token_dim=256, img_hw=(540,960), reduce_ch=None):
        super().__init__()
        self.encoder = CameraEncoder(token_dim, img_hw, reduce_ch=reduce_ch)
        self.fusion  = nn.Linear(256, embed_dim)


    def forward(self, images):
        e, t = self.encoder(images[:, 0])
        return self.fusion(e), t


class LidarEncoder(nn.Module):
    """PointNet features -> (dense (B, embed_dim), tokens (B, M, token_dim)).

    Dense head is an attention-pool + max-pool design. Token head is PMA
    (pooling by multi-head attention, Set-Transformer style): M learnable
    seeds cross-attend over the valid points, giving M independent
    geometric summaries instead of one pooled vector.
    """
    def __init__(self, embed_dim=128, token_dim=256, num_tokens=16, num_heads=8,
                 dropout=0.05):
        super().__init__()
        self.num_tokens = num_tokens
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64),   nn.GELU(),
            nn.Linear(64, 128), nn.GELU(),
            nn.Linear(128, 256), nn.LayerNorm(256), nn.GELU(),
        )
        # dense head
        self.attention_mlp = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 64),  nn.GELU(),
            nn.Linear(64, 1),
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, embed_dim),
        )
        # token head (PMA)
        self.to_tok = nn.Linear(256, token_dim)
        self.seeds  = nn.Parameter(torch.randn(1, num_tokens, token_dim) * 0.02)
        self.pma    = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)
        self.norm   = nn.LayerNorm(token_dim)
        self.ffn    = nn.Sequential(nn.Linear(token_dim, token_dim * 2), nn.GELU(),
                                    nn.Linear(token_dim * 2, token_dim))

    def forward(self, points):
        B, N, _ = points.shape
        valid = (points.abs().sum(-1) > 1e-6)                                # (B,N)
        no_valid = ~valid.any(dim=1)
        if no_valid.any():
            valid = valid.clone()
            valid[no_valid, 0] = True

        pf = self.point_mlp(points)                                          # (B,N,256)

        scores = self.attention_mlp(pf).squeeze(-1)                          # (B,N)
        scores = scores.masked_fill(~valid, float('-inf'))
        w = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0).unsqueeze(-1)
        attn_feat = (pf * w).sum(1)
        pf_m = pf.masked_fill(~valid.unsqueeze(-1), float('-inf'))
        max_feat = torch.nan_to_num(pf_m.max(1)[0], nan=0.0, neginf=0.0)
        emb = self.output_mlp(torch.cat([attn_feat, max_feat], -1))          # (B,embed_dim)

        kv = self.to_tok(pf)                                                 # (B,N,d)
        q  = self.seeds.expand(B, -1, -1).to(kv.dtype)
        out, _ = self.pma(q, kv, kv, key_padding_mask=~valid)
        out = self.norm(out)
        out = out + self.ffn(out)
        return emb, torch.nan_to_num(out, nan=0.0)                           # (B,M,d)


print("[OK] Dual-path encoders defined "
      "(v5: dense full-resolution CSI readout restored + token sequences)")

class PhysicsAnchor(nn.Module):
    """Differentiable beamspace peak reader -> NUM_ANCHOR_STATS statistics
    (12 for a ULA, 14 for a UPA) + coarse prior.

    Runs in fp32 regardless of AMP: the softmax over N_ant*N_sub log-power
    bins and the entropy term are precision-sensitive, and a silent fp16
    overflow inside a probabilistic head was a real failure mode in an
    earlier version.
    """
    def __init__(self, num_ant=16, num_sub=8, hidden=128, out_dim=3,
                 is_upa=None, ant_rows=None, ant_cols=None):
        super().__init__()
        self.n_bins = num_ant * num_sub
        self.is_upa = IS_UPA if is_upa is None else is_upa
        self.num_anchor_stats = 14 if self.is_upa else 12

        if self.is_upa:
            rows = ANT_ROWS if ant_rows is None else ant_rows
            cols = ANT_COLS if ant_cols is None else ant_cols
            assert rows * cols == num_ant, \
                f"ant_rows*ant_cols ({rows}*{cols}) must equal num_ant ({num_ant})"
            idx = torch.arange(num_ant)
            row_of_idx = idx // cols
            col_of_idx = idx % cols
            row_coord = torch.linspace(-1., 1., rows)[row_of_idx]            # (num_ant,)
            col_coord = torch.linspace(-1., 1., cols)[col_of_idx]            # (num_ant,)
            gax = col_coord.view(num_ant, 1).expand(num_ant, num_sub)        # azimuth   (cols)
            gay = row_coord.view(num_ant, 1).expand(num_ant, num_sub)        # elevation (rows)
            self.register_buffer('grid_ax', gax.reshape(1, -1).contiguous())
            self.register_buffer('grid_ay', gay.reshape(1, -1).contiguous())
        else:
            ga = torch.linspace(-1., 1., num_ant).view(num_ant, 1).expand(num_ant, num_sub)
            self.register_buffer('grid_a', ga.reshape(1, -1).contiguous())

        gs = torch.linspace(-1., 1., num_sub).view(1, num_sub).expand(num_ant, num_sub)
        self.register_buffer('grid_s', gs.reshape(1, -1).contiguous())
        self.log_beta   = nn.Parameter(torch.tensor(math.log(4.0)))
        self.anchor_mlp = nn.Sequential(
            nn.Linear(self.num_anchor_stats, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        nn.init.zeros_(self.anchor_mlp[-1].weight)
        nn.init.zeros_(self.anchor_mlp[-1].bias)

    def forward(self, csi_feat):
        dev = 'cuda' if csi_feat.is_cuda else 'cpu'
        with torch.autocast(device_type=dev, enabled=False):
            f = csi_feat.float()
            fftlog = f[:, 5].flatten(1)                                      # (B, n_bins)
            logmag = f[:, 2].flatten(1)

            fmean = fftlog.mean(dim=1, keepdim=True)
            fstd  = fftlog.std(dim=1, keepdim=True).clamp_min(1e-6)
            q = (fftlog - fmean) / fstd

            beta = torch.exp(self.log_beta).clamp(0.5, 50.0)
            p = torch.softmax(beta * q, dim=1)                               # (B, n_bins) --
                                                                               # same angle-delay dist either way;
                                                                               # only the readout below differs
            gs  = self.grid_s.to(p.dtype)
            u_s  = (p * gs).sum(1, keepdim=True)
            sd_s = ((p * (gs - u_s) ** 2).sum(1, keepdim=True)).clamp_min(0.).sqrt()

            if self.is_upa:
                gax = self.grid_ax.to(p.dtype)
                gay = self.grid_ay.to(p.dtype)
                u_ax  = (p * gax).sum(1, keepdim=True)                       # soft-argmax azimuth
                u_ay  = (p * gay).sum(1, keepdim=True)                       # soft-argmax elevation
                sd_ax = ((p * (gax - u_ax) ** 2).sum(1, keepdim=True)).clamp_min(0.).sqrt()
                sd_ay = ((p * (gay - u_ay) ** 2).sum(1, keepdim=True)).clamp_min(0.).sqrt()
                angle_stats = [u_ax, u_ay, u_s, sd_ax, sd_ay, sd_s]          # 6 stats
            else:
                ga  = self.grid_a.to(p.dtype)
                u_a  = (p * ga).sum(1, keepdim=True)                         # soft-argmax
                sd_a = ((p * (ga - u_a) ** 2).sum(1, keepdim=True)).clamp_min(0.).sqrt()
                angle_stats = [u_a, u_s, sd_a, sd_s]                         # 4 stats

            conc_max = p.max(dim=1, keepdim=True).values                     # peak mass
            ent = -(p * torch.log(p + 1e-12)).sum(1, keepdim=True) / math.log(self.n_bins)
            conc_ent = 1.0 - ent                                             # K-factor proxy

            f_top  = fftlog.max(dim=1, keepdim=True).values
            f_mean = fmean
            f_std  = fstd
            m_mean = logmag.mean(dim=1, keepdim=True)                        # path loss / range
            m_std  = logmag.std(dim=1, keepdim=True)
            m_top  = logmag.max(dim=1, keepdim=True).values

            stats = torch.cat(angle_stats + [conc_max, conc_ent,
                               f_top, f_mean, f_std, m_mean, m_std, m_top], dim=1)
            assert stats.shape[1] == self.num_anchor_stats, \
                f"PhysicsAnchor produced {stats.shape[1]} stats, expected self.num_anchor_stats={self.num_anchor_stats}"
            stats = torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)
            delta0 = self.anchor_mlp(stats)                                  # (B, out_dim)
        return delta0, stats


class FusionLayer(nn.Module):
    """One pre-norm gated-bottleneck layer: latent tokens Z cross-attend to
    each modality with residual connections, each contribution scaled by a
    per-sample reliability gate r_m in (0,1); then latent self-attention;
    then an FFN. Residual branches carry LayerScale so the layer starts near
    the identity."""
    def __init__(self, dim, num_heads, mod_names, dropout=0.05, ffn_mult=4,
                 ls_init=1e-4):
        super().__init__()
        self.mod_names = list(mod_names)
        self.q_norms = nn.ModuleDict({m: nn.LayerNorm(dim) for m in self.mod_names})
        self.cross   = nn.ModuleDict({m: nn.MultiheadAttention(dim, num_heads,
                                       dropout=dropout, batch_first=True)
                                      for m in self.mod_names})
        self.rel_gate = nn.ModuleDict({m: nn.Linear(2 * dim, 1) for m in self.mod_names})
        for m in self.mod_names:                       # start ~open (sigmoid(1)=0.73)
            nn.init.zeros_(self.rel_gate[m].weight)
            nn.init.constant_(self.rel_gate[m].bias, 1.0)
        self.ls_cross = nn.ParameterDict(
            {m: nn.Parameter(torch.full((dim,), ls_init)) for m in self.mod_names})

        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                               batch_first=True)
        self.ls_self   = nn.Parameter(torch.full((dim,), ls_init))
        self.ffn_norm  = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
        )
        self.ls_ffn = nn.Parameter(torch.full((dim,), ls_init))

    def forward(self, Z, tokens):
        gates = {}
        for m in self.mod_names:
            if m not in tokens:                        # disabled or dropped this step
                continue
            E = tokens[m]
            r = torch.sigmoid(self.rel_gate[m](
                torch.cat([Z.mean(1), E.mean(1)], dim=-1)))                  # (B,1)
            attn, _ = self.cross[m](self.q_norms[m](Z), E, E)
            Z = Z + r.unsqueeze(1) * self.ls_cross[m].to(attn.dtype) * attn
            gates[m] = r
        zn = self.self_norm(Z)
        Z = Z + self.ls_self.to(Z.dtype) * self.self_attn(zn, zn, zn)[0]
        Z = Z + self.ls_ffn.to(Z.dtype) * self.ffn(self.ffn_norm(Z))
        return Z, gates


class GPSLocalizer(nn.Module):
    
    def __init__(self, *,
                 use_csi=True, use_bs_pos=True, use_camera=False, use_lidar=False,
                 use_all_cameras=False, num_cameras=4,
                 csi_embed_dim=384, pos_embed_dim=96,
                 cam_embed_dim=192, lidar_embed_dim=128,
                 fusion_dim=768, dropout=0.05,
                 csi_width=96, csi_blocks=4, csi_flatten_ch=None, cam_reduce_ch=None,
                 num_ant=16, num_sub=8, csi_in_ch=12, out_dim=3,
                 img_hw=(540,960),
                 use_attn_refine=True, attn_dim=256, num_heads=8,
                 num_fusion_tokens=8, num_fusion_layers=3, num_lidar_tokens=16,
                 use_dist_head=True, num_bins=128, bin_range=5.0,
                 axis_scale=None, mod_drop_p=0.0, ls_init=1e-4,
                 use_csi_confidence_gate=True, use_cam_confidence_gate=True,
                 is_upa=None, ant_rows=None, ant_cols=None):
        super().__init__()
        self.use_csi, self.use_bs_pos = use_csi, use_bs_pos
        self.use_camera, self.use_lidar = use_camera, use_lidar
        self.use_attn_refine = use_attn_refine
        self.use_dist_head   = use_dist_head
        self.out_dim    = out_dim
        self.num_bins   = num_bins
        if isinstance(mod_drop_p, dict):
            self.mod_drop_p = dict(mod_drop_p)
        else:
            self.mod_drop_p = {'csi': float(mod_drop_p), 'cam': float(mod_drop_p),
                               'lidar': float(mod_drop_p)}
        # antenna geometry, defaulting to the global config but overridable
        # so different instances can target different ANT_SHAPE values
        self.is_upa   = IS_UPA  if is_upa  is None else is_upa
        self.ant_rows = ANT_ROWS if ant_rows is None else ant_rows
        self.ant_cols = ANT_COLS if ant_cols is None else ant_cols

        mods, fusion_in = [], 0
        if use_csi:
            self.csi_encoder = CSIEncoder(csi_embed_dim, attn_dim, num_ant, num_sub,
                                          in_ch=csi_in_ch, width=csi_width,
                                          num_blocks=csi_blocks, dropout=dropout,
                                          flatten_ch=csi_flatten_ch,
                                          is_upa=self.is_upa, ant_rows=self.ant_rows,
                                          ant_cols=self.ant_cols)
            self.csi_norm = nn.LayerNorm(csi_embed_dim)
            self.anchor   = PhysicsAnchor(num_ant, num_sub, out_dim=out_dim,
                                          is_upa=self.is_upa, ant_rows=self.ant_rows,
                                          ant_cols=self.ant_cols)
            self.num_anchor_stats = self.anchor.num_anchor_stats
            fusion_in += csi_embed_dim + self.num_anchor_stats + out_dim
            mods.append('csi')
        if use_camera:
            self.cam_encoder = (
                MultiCameraEncoder(num_cameras, cam_embed_dim, attn_dim, img_hw,
                                   reduce_ch=cam_reduce_ch)
                if use_all_cameras
                else SingleCameraEncoder(cam_embed_dim, attn_dim, img_hw,
                                         reduce_ch=cam_reduce_ch))
            self.cam_norm = nn.LayerNorm(cam_embed_dim)
            fusion_in += cam_embed_dim
            mods.append('cam')
        if use_lidar:
            self.lidar_encoder = LidarEncoder(lidar_embed_dim, attn_dim,
                                              num_lidar_tokens, num_heads, dropout)
            self.lidar_norm = nn.LayerNorm(lidar_embed_dim)
            fusion_in += lidar_embed_dim
            mods.append('lidar')
        if use_bs_pos:
            self.bs_pos_encoder = BSPosEncoder(pos_embed_dim, attn_dim)
            self.bs_pos_norm    = nn.LayerNorm(pos_embed_dim)
            fusion_in += pos_embed_dim
            mods.append('bs')
        assert len(mods) > 0, "Enable at least one modality!"
        self.mod_names = mods

        miss = {}
        if use_csi and self.mod_drop_p.get('csi', 0.0) > 0:
            miss['csi']   = nn.Parameter(torch.zeros(csi_embed_dim))
        if use_camera and self.mod_drop_p.get('cam', 0.0) > 0:
            miss['cam']   = nn.Parameter(torch.zeros(cam_embed_dim))
        if use_lidar and self.mod_drop_p.get('lidar', 0.0) > 0:
            miss['lidar'] = nn.Parameter(torch.zeros(lidar_embed_dim))
        self.missing_emb = nn.ParameterDict(miss)

        self.use_csi_gate = use_csi and use_camera and use_csi_confidence_gate
        if self.use_csi_gate:
            self.csi_conf_gate = nn.Linear(csi_embed_dim + cam_embed_dim, 1)
            nn.init.zeros_(self.csi_conf_gate.weight)
            nn.init.constant_(self.csi_conf_gate.bias, 1.0)

        self.use_cam_gate = use_csi and use_camera and use_cam_confidence_gate
        if self.use_cam_gate:
            self.cam_conf_gate = nn.Linear(csi_embed_dim + cam_embed_dim, 1)
            nn.init.zeros_(self.cam_conf_gate.weight)
            nn.init.constant_(self.cam_conf_gate.bias, 1.0)

        self.trunk = nn.Sequential(
            nn.Linear(fusion_in, fusion_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim), nn.GELU(), nn.Dropout(dropout),
        )

        if use_attn_refine:
            self.fusion_tokens = nn.Parameter(
                torch.randn(1, num_fusion_tokens, attn_dim) * 0.02)
            cond_dim = (self.num_anchor_stats + out_dim) if use_csi else 1
            self.film = nn.Sequential(nn.Linear(cond_dim, 128), nn.GELU(),
                                      nn.Linear(128, 2 * attn_dim))
            nn.init.zeros_(self.film[-1].weight); nn.init.zeros_(self.film[-1].bias)
            self.layers = nn.ModuleList([
                FusionLayer(attn_dim, num_heads, mods, dropout=dropout,
                           ls_init=ls_init)
                for _ in range(num_fusion_layers)
            ])
            self.refine_norm = nn.LayerNorm(attn_dim)
            self.refine_proj = nn.Linear(num_fusion_tokens * attn_dim, fusion_dim)
            self.refine_scale = nn.Parameter(torch.zeros(fusion_dim))

        self.head_reg = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, out_dim),      # no activation -> can predict negatives
        )
        if use_dist_head:
            nn.init.zeros_(self.head_reg[-1].weight)
            nn.init.zeros_(self.head_reg[-1].bias)
            self.head_dist = nn.Sequential(
                nn.Linear(fusion_dim, 256), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(256, out_dim * num_bins),
            )
            if axis_scale is None:
                axis_scale = np.ones(out_dim, dtype=np.float32)
            a = torch.as_tensor(np.asarray(axis_scale, dtype=np.float32)[:out_dim])
            grid = torch.linspace(-bin_range, bin_range, num_bins)           # (K,)
            self.register_buffer('bin_centers',
                                 (a.view(out_dim, 1) * grid.view(1, num_bins)).contiguous())

    def _encode(self, features):
        """-> (dense dict, token dict, anchor_pred or None, stats or None)"""
        dense, tokens = {}, {}
        anchor_pred, stats = None, None
        if self.use_csi:
            e, t = self.csi_encoder(features['csi_feat'])
            dense['csi'], tokens['csi'] = self.csi_norm(e), t
            anchor_pred, stats = self.anchor(features['csi_feat'])
        if self.use_camera:
            e, t = self.cam_encoder(features['images'])
            dense['cam'], tokens['cam'] = self.cam_norm(e), t
        if self.use_lidar:
            e, t = self.lidar_encoder(features['lidar'])
            dense['lidar'], tokens['lidar'] = self.lidar_norm(e), t
        if self.use_bs_pos:
            e, t = self.bs_pos_encoder(features['bs_pos'])
            dense['bs'], tokens['bs'] = self.bs_pos_norm(e), t
        return dense, tokens, anchor_pred, stats

    def forward(self, features, force_drop=None, disable_random_dropout=False,
                return_embedding=False):
        dense, tokens, anchor_pred, stats = self._encode(features)

        csi_dropped = False
        cam_dropped = False
        if force_drop:
            for m in force_drop:
                if m in dense and len(dense) > 1:
                    if m not in self.missing_emb:
                        raise KeyError(
                            f"force_drop requested '{m}' but this checkpoint has no "
                            f"missing_emb['{m}'] -- it was trained with "
                            f"MOD_DROP_P['{m}']==0, so there is no learned substitute "
                            f"to use. Ablation is only meaningful for a modality the "
                            f"model was actually trained to sometimes do without.")
                    B = dense[m].size(0)
                    dense[m] = self.missing_emb[m].to(dense[m].dtype).expand(B, -1)
                    tokens.pop(m, None)
                    if m == 'csi':
                        csi_dropped = True
                    elif m == 'cam':
                        cam_dropped = True
        if self.training and not disable_random_dropout:
            for m in ('csi', 'cam', 'lidar'):
                p = self.mod_drop_p.get(m, 0.0)
                if (p > 0 and m in dense and len(dense) > 1
                        and torch.rand(()).item() < p):
                    B = dense[m].size(0)
                    dense[m] = self.missing_emb[m].to(dense[m].dtype).expand(B, -1)
                    tokens.pop(m, None)
                    if m == 'csi':
                        csi_dropped = True
                    elif m == 'cam':
                        cam_dropped = True

        csi_confidence = None
        cam_confidence = None
        gate_diag = {}
        
        _joint_ctx = None
        if 'csi' in dense and 'cam' in dense:
            _joint_ctx = torch.cat([dense['csi'], dense['cam']], dim=-1).float()

        if self.use_csi_gate and not csi_dropped and _joint_ctx is not None:
            csi_confidence = torch.sigmoid(
                self.csi_conf_gate(_joint_ctx).to(dense['csi'].dtype))   # (B,1)
        if self.use_cam_gate and not cam_dropped and _joint_ctx is not None:
            cam_confidence = torch.sigmoid(
                self.cam_conf_gate(_joint_ctx).to(dense['cam'].dtype))   # (B,1)

        if csi_confidence is not None:
            dense['csi'] = dense['csi'] * csi_confidence
            gate_diag['csi_confidence'] = [csi_confidence]
        if cam_confidence is not None:
            dense['cam'] = dense['cam'] * cam_confidence
            gate_diag['cam_confidence'] = [cam_confidence]

        embs = [dense[m] for m in self.mod_names if m in dense]
        ref_dtype = embs[0].dtype
        if stats is not None:
            stats_for_trunk, anchor_for_trunk = stats, anchor_pred
            if csi_dropped:
                stats = torch.zeros_like(stats)
                anchor_pred = torch.zeros_like(anchor_pred)
                stats_for_trunk, anchor_for_trunk = stats, anchor_pred
            elif csi_confidence is not None:
                stats_for_trunk = stats * csi_confidence
                anchor_for_trunk = anchor_pred * csi_confidence
            embs.append(stats_for_trunk.to(ref_dtype))
            embs.append(anchor_for_trunk.to(ref_dtype))
        h = self.trunk(torch.cat(embs, dim=1))

        if self.use_attn_refine:
            B = h.size(0)
            Z = self.fusion_tokens.expand(B, -1, -1).to(ref_dtype)
            if stats is not None:
                cond = torch.cat([stats, anchor_pred], dim=1).to(Z.dtype)
            else:
                cond = torch.zeros(B, 1, device=Z.device, dtype=Z.dtype)
            gamma, beta = self.film(cond).chunk(2, dim=-1)
            Z = Z * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
            for layer in self.layers:
                Z, g = layer(Z, tokens)
                for k, v in g.items():
                    gate_diag.setdefault(k, []).append(v)
            z = self.refine_norm(Z).flatten(1)          # concat the K latents,
            z = self.refine_proj(z)                     # not mean-pooled
            h = h + self.refine_scale.to(h.dtype) * z

        delta_res = self.head_reg(h)
        bin_logits = None
        if self.use_dist_head:
            bin_logits = self.head_dist(h).view(-1, self.out_dim, self.num_bins)
            p_bin = torch.softmax(bin_logits.float(), dim=-1)
            mu = (p_bin * self.bin_centers.unsqueeze(0)).sum(-1)              # (B,out_dim)
            pred = mu.to(delta_res.dtype) + delta_res
        else:
            pred = delta_res
        if return_embedding:
            emb = next(iter(dense.values())) if dense else None
            return pred, anchor_pred, bin_logits, gate_diag, emb
        return pred, anchor_pred, bin_logits, gate_diag


print("[OK] GPSLocalizer defined (v5 DFR: dense trunk + repaired physics anchor "
      "+ zero-init attention residual + distributional head)")


# ---------------------------------------------------------------------------
# CADPool -- Conflict-Aware Distributional Pooling
# ---------------------------------------------------------------------------
#
# 1. pool in distribution space: log-opinion pool, p_fused ~ p_csi^a *
#    p_cam^b, in log space -- preserves multi-modality instead of averaging
#    means into a point neither expert believes in.
# 2. per-sample, per-axis exponents (a, b are (B, out_dim)), since camera
#    and CSI are strong/weak on different axes for different reasons.
# 3. exponents are conflict-aware: the descriptor MLP sees each expert's
#    mean/spread plus their disagreement (JS divergence + standardized mean
#    gap), since "confident and disagree" needs different handling than
#    "confident and agree."
# 4. zero-init last layer: at epoch 0 the exponents collapse to the
#    inverse-variance (Kalman-style) precision weights, so training starts
#    from the classical closed-form fusion and can only learn departures.
class CADPool(nn.Module):
    """Conflict-aware distributional pooling of two frozen experts.

    Consumes each expert's point prediction and per-axis bin logits; returns
    a fused point prediction plus fused bin logits (so the existing DFL term
    in DFRLoss still applies to the fused distribution, not just the point).
    """

    N_DESC = 10          # descriptors per axis (see _descriptors)

    def __init__(self, out_dim, num_bins, bin_centers, csi_embed_dim, cam_embed_dim,
                 emb_hidden=16, hidden=64, dropout=0.05,
                 mod_drop_p=0.15, gate_clamp=3.0, corr_scale=0.10):
        super().__init__()
        self.out_dim    = int(out_dim)
        self.num_bins   = int(num_bins)
        self.mod_drop_p = float(mod_drop_p)
        self.gate_clamp = float(gate_clamp)
        self.corr_scale = float(corr_scale)
        self.register_buffer('bin_centers', bin_centers.clone().float())  # (D,K)

        self.emb_compress_csi = nn.Sequential(
            nn.Linear(int(csi_embed_dim), emb_hidden), nn.GELU())
        self.emb_compress_cam = nn.Sequential(
            nn.Linear(int(cam_embed_dim), emb_hidden), nn.GELU())

        in_dim  = self.out_dim * self.N_DESC + 2 * emb_hidden
        out_n   = 3 * self.out_dim                 # g_csi(D), g_cam(D), corr(D)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_n),
        )
        self._zero_init()

    def _zero_init(self):
        """Last layer -> exactly zero, so the module starts as the classical
        precision-weighted log-opinion pool (see header, point 4)."""
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _moments(self, logits):
        """logits (B,D,K) -> log p, p, mean (B,D), sd (B,D), norm-entropy (B,D)."""
        logp = torch.log_softmax(logits.float(), dim=-1)
        p    = logp.exp()
        c    = self.bin_centers.unsqueeze(0)                       # (1,D,K)
        mu   = (p * c).sum(-1)                                     # (B,D)
        var  = (p * (c - mu.unsqueeze(-1)) ** 2).sum(-1).clamp_min(1e-8)
        ent  = -(p * logp).sum(-1) / math.log(self.num_bins)       # (B,D) in [0,1]
        return logp, p, mu, var.sqrt(), ent

    def _descriptors(self, mu_a, sd_a, ent_a, res_a, mu_b, sd_b, ent_b, res_b,
                     p_a, p_b):
        """The compact, physically-meaningful feature vector the gate sees.

        Deliberately not the raw expert embeddings: with ~5k training
        samples a combiner that large would overfit, and freezing the
        experts means the representation learning is already done. Ten
        scalars per axis is enough to say who's sharper, who's more
        certain, and whether they agree.
        """
        pa_c   = p_a.clamp_min(1e-12)
        pb_c   = p_b.clamp_min(1e-12)
        m      = (0.5 * (pa_c + pb_c)).clamp_min(1e-12)
        js     = 0.5 * ((pa_c * (pa_c / m).log()).sum(-1)
                        + (pb_c * (pb_c / m).log()).sum(-1))
        js     = (js / math.log(2.0)).clamp(0.0, 1.0)               # (B,D) in [0,1]
        dmu    = (mu_a - mu_b) / (sd_a.pow(2) + sd_b.pow(2)).clamp_min(1e-8).sqrt()
        desc   = torch.stack([mu_a, mu_b, sd_a, sd_b, ent_a, ent_b,
                              js, dmu, res_a, res_b], dim=-1)       # (B,D,10)
        return desc, js

    def forward(self, pred_a, logits_a, pred_b, logits_b, emb_a, emb_b,
                force_drop=None, disable_random_dropout=False):
        """a == CSI expert, b == camera expert. All inputs already detached."""
        logp_a, p_a, mu_a, sd_a, ent_a = self._moments(logits_a)
        logp_b, p_b, mu_b, sd_b, ent_b = self._moments(logits_b)

        res_a = pred_a.float() - mu_a
        res_b = pred_b.float() - mu_b

        desc, js = self._descriptors(mu_a, sd_a, ent_a, res_a,
                                     mu_b, sd_b, ent_b, res_b, p_a, p_b)
        B = desc.size(0)
        emb_feat_a = self.emb_compress_csi(emb_a.float())   # (B, emb_hidden)
        emb_feat_b = self.emb_compress_cam(emb_b.float())   # (B, emb_hidden)
        net_in = torch.cat([desc.reshape(B, -1), emb_feat_a, emb_feat_b], dim=-1)
        out   = self.net(net_in)                                    # (B, 3D)
        g_a, g_b, corr = out.split(self.out_dim, dim=-1)

        prec_a = 1.0 / sd_a.pow(2).clamp_min(1e-8)
        prec_b = 1.0 / sd_b.pow(2).clamp_min(1e-8)
        w_a    = prec_a / (prec_a + prec_b)          # reported, not applied

        c = self.gate_clamp
        alpha_a = torch.exp(g_a.clamp(-c, c))
        alpha_b = torch.exp(g_b.clamp(-c, c))

        drop_a = drop_b = False
        if force_drop:
            drop_a = 'csi' in force_drop
            drop_b = 'cam' in force_drop
            if drop_a and drop_b:        # never drop both -- nothing would remain
                drop_a = drop_b = False
        if self.training and not disable_random_dropout and self.mod_drop_p > 0:
            if not (drop_a or drop_b):
                r = torch.rand(()).item()
                if r < self.mod_drop_p:
                    drop_a = True
                elif r < 2 * self.mod_drop_p:
                    drop_b = True
        if drop_a:
            alpha_a = torch.zeros_like(alpha_a)
            alpha_b = torch.ones_like(alpha_b)
        elif drop_b:
            alpha_b = torch.zeros_like(alpha_b)
            alpha_a = torch.ones_like(alpha_a)

        fused_logits = (alpha_a.unsqueeze(-1) * logp_a
                        + alpha_b.unsqueeze(-1) * logp_b)           # (B,D,K)
        p_f  = torch.softmax(fused_logits, dim=-1)
        mu_f = (p_f * self.bin_centers.unsqueeze(0)).sum(-1)        # (B,D)

        eff_a = alpha_a * prec_a
        eff_b = alpha_b * prec_b
        esum  = (eff_a + eff_b).clamp_min(1e-8)
        res_f = (eff_a * res_a + eff_b * res_b) / esum
        pred  = mu_f + res_f + self.corr_scale * torch.tanh(corr)

        a_n  = eff_a / esum                  # effective share, comparable to a gate
        diag = {'csi': [a_n.detach()],
                'cam': [(1.0 - a_n).detach()],
                'conflict': [js.detach()],
                'prec_csi': [w_a.detach()]}
        return pred, fused_logits, diag


class LateFusionLocalizer(nn.Module):
    """Drop-in replacement for GPSLocalizer in a two-modality run.

    Exposes the exact same forward contract --
        (features, force_drop, disable_random_dropout)
            -> (pred, anchor_pred, bin_logits, gate_diag)
    -- so every downstream systen (training loop, EMA, TTA, evaluate_full,
    results.mat/.txt/CDF/curves) works unchanged. The benchmark harness must
    not vary between the two fusion modes, or the A/B comparison it exists
    to support would be meaningless.

    anchor_pred is returned as None: the physics anchor is a deep-
    supervision signal on the CSI encoder, which is frozen here, so there's
    nothing left for that gradient to teach. DFRLoss already skips the
    anchor term on None.
    """

    def __init__(self, csi_model, cam_model, out_dim, num_bins,
                 csi_embed_dim, cam_embed_dim, emb_hidden=16,
                 hidden=64, dropout=0.05, mod_drop_p=0.15,
                 gate_clamp=3.0, corr_scale=0.10):
        super().__init__()
        if not (hasattr(csi_model, 'bin_centers') and hasattr(cam_model, 'bin_centers')):
            raise ValueError("[late fusion] both experts need a distributional head "
                             "(USE_DIST_HEAD=True) -- CADPool fuses distributions.")
        if not torch.allclose(csi_model.bin_centers.float().cpu(),
                              cam_model.bin_centers.float().cpu(), atol=1e-6):
            raise ValueError(
                "[late fusion] the two experts do not share a bin grid. A log-"
                "opinion pool is only defined on a COMMON support -- retrain "
                "both base models with identical AXIS_SCALE/BIN_RANGE/NUM_BINS.")
        self.csi_model = csi_model
        self.cam_model = cam_model
        for m in (self.csi_model, self.cam_model):
            for p in m.parameters():
                p.requires_grad_(False)
        self.pool = CADPool(
            out_dim=out_dim, num_bins=num_bins,
            bin_centers=csi_model.bin_centers.detach().cpu(),
            csi_embed_dim=csi_embed_dim, cam_embed_dim=cam_embed_dim,
            emb_hidden=emb_hidden,
            hidden=hidden, dropout=dropout, mod_drop_p=mod_drop_p,
            gate_clamp=gate_clamp, corr_scale=corr_scale)
        self.register_buffer('bin_centers', csi_model.bin_centers.detach().clone())

        self.mod_names   = ['csi', 'cam']
        self.missing_emb = {'csi': True, 'cam': True}

    def train(self, mode: bool = True):
        """Keep the frozen experts in eval() always.

        Without this override, model.train() would flip the experts'
        dropout and BatchNorm back into training behaviour every epoch.
        Their weights wouldn't move (requires_grad=False), but their
        outputs would become stochastic and their BN running stats would
        drift on the training data -- the combiner would be fitting a
        moving target and test-time behaviour would silently diverge from
        what was validated.
        """
        super().train(mode)
        self.csi_model.eval()
        self.cam_model.eval()
        return self

    def forward(self, features, force_drop=None, disable_random_dropout=False):
        with torch.no_grad():                       # experts are frozen: no graph
            pred_a, _, logits_a, _, emb_a = self.csi_model(features, return_embedding=True)
            pred_b, _, logits_b, _, emb_b = self.cam_model(features, return_embedding=True)
        if logits_a is None or logits_b is None:
            raise RuntimeError("[late fusion] an expert returned no bin logits; "
                               "both base models must have USE_DIST_HEAD=True.")
        pred, fused_logits, diag = self.pool(
            pred_a.detach().float(), logits_a.detach().float(),
            pred_b.detach().float(), logits_b.detach().float(),
            emb_a.detach().float(), emb_b.detach().float(),
            force_drop=force_drop, disable_random_dropout=disable_random_dropout)
        return pred, None, fused_logits, diag


print("[OK] CADPool + LateFusionLocalizer defined (v16: conflict-aware "
      "distributional log-opinion pooling, zero-init at the precision-weighted "
      "classical estimator; v24: gate also sees each expert's compressed "
      "penultimate embedding, previously computed and discarded)")

def get_cosine_warmup_scheduler(optimizer, warmup_epochs, total_epochs,
                                 min_lr=1e-7, base_lr=1e-3):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            alpha = epoch / max(1, warmup_epochs)
            return (min_lr / base_lr) + alpha * (1.0 - min_lr / base_lr)
        prog = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return (min_lr / base_lr) + 0.5 * (1.0 - min_lr / base_lr) * (1 + math.cos(math.pi * prog))
    return LambdaLR(optimizer, lr_lambda)


def beta_at_epoch(epoch, total_epochs, beta_start, beta_end):
    """Cosine anneal of the SmoothL1 transition point (1-indexed epochs)."""
    prog = min(max((epoch - 1) / max(1, total_epochs - 1), 0.0), 1.0)
    return beta_end + 0.5 * (beta_start - beta_end) * (1.0 + math.cos(math.pi * prog))


class OffsetLoss(nn.Module):
    """SmoothL1_beta(pred, target) + lambda * E[ ||pred - target||_2 ].

    `beta` is mutable so the training loop can anneal it (see set_beta)."""
    def __init__(self, beta=0.30, lambda_dist=1.0):
        super().__init__()
        self.beta = float(beta)
        self.lambda_dist = float(lambda_dist)

    def set_beta(self, beta):
        self.beta = float(max(beta, 1e-6))

    def forward(self, pred, target):
        l_huber = F.smooth_l1_loss(pred.float(), target.float(), beta=self.beta)
        l_dist  = torch.norm(pred.float() - target.float(), dim=1).mean()
        return l_huber + self.lambda_dist * l_dist


def distribution_focal_loss(bin_logits, target, bin_centers):
    """DFL: two-bin linearly-interpolated cross-entropy.

    Places the target mass on the two bin centres that straddle the true
    value, in proportion to proximity, so the argmax-free expectation of the
    learned distribution is an unbiased estimator of the target. Computed in
    fp32; targets outside the grid are clamped to its edge bins.

      bin_logits : (B, D, K)   raw logits
      target     : (B, D)      scaled offset
      bin_centers: (D, K)      uniformly spaced ascending centres per axis
    """
    logits = bin_logits.float()
    tgt    = target.float()
    D, K = bin_centers.shape
    lo   = bin_centers[:, 0].unsqueeze(0)                                    # (1,D)
    step = (bin_centers[:, 1] - bin_centers[:, 0]).clamp_min(1e-8).unsqueeze(0)

    t  = ((tgt - lo) / step).clamp(0.0, K - 1 - 1e-4)                        # (B,D)
    il = t.floor().long()
    ir = (il + 1).clamp_max(K - 1)
    wr = (t - il.to(t.dtype))
    wl = 1.0 - wr

    logp = F.log_softmax(logits, dim=-1)                                     # (B,D,K)
    ll = logp.gather(-1, il.unsqueeze(-1)).squeeze(-1)
    lr = logp.gather(-1, ir.unsqueeze(-1)).squeeze(-1)
    return -(wl * ll + wr * lr).mean()


class DFRLoss(nn.Module):
    """Total loss: metric-aligned regression + distributional head + deep
    supervision on the physics anchor. Every term is a regression or
    proper-scoring term on the same target, so nothing here can trade mean
    accuracy for inflated uncertainty."""
    def __init__(self, beta=0.30, lambda_dist=1.0, lambda_dfl=0.25,
                 lambda_anchor=0.10, bin_centers=None):
        super().__init__()
        self.main   = OffsetLoss(beta, lambda_dist)
        self.anchor = OffsetLoss(beta, lambda_dist)
        self.lambda_dfl    = float(lambda_dfl)
        self.lambda_anchor = float(lambda_anchor)
        self.bin_centers   = bin_centers      # torch tensor (D, K) or None

    def set_beta(self, beta):
        self.main.set_beta(beta)
        self.anchor.set_beta(beta)

    @property
    def beta(self):
        return self.main.beta

    def forward(self, pred, anchor_pred, bin_logits, target):
        loss = self.main(pred, target)
        if bin_logits is not None and self.lambda_dfl > 0 and self.bin_centers is not None:
            loss = loss + self.lambda_dfl * distribution_focal_loss(
                bin_logits, target, self.bin_centers.to(bin_logits.device))
        if anchor_pred is not None and self.lambda_anchor > 0:
            loss = loss + self.lambda_anchor * self.anchor(anchor_pred, target)
        return loss


class ModelEMA:
    """Exponential moving average of model parameters (and buffers), with
    bias correction. The EMA weights -- not the raw SGD iterates -- are
    validated, checkpointed and used at test time.

    The effective decay ramps as decay * (1 - exp(-n / tau)). With a fixed
    decay the shadow would spend its first few thousand steps dominated by
    the random initialization, making early validation numbers meaningless
    and early-stopping bookkeeping start from noise. The ramp makes the EMA
    track the model closely at first and lengthen its horizon as training
    settles."""
    def __init__(self, model, decay=0.9995, warmup_steps=2000):
        self.decay = decay
        self.warmup_steps = max(int(warmup_steps), 1)
        self.num_updates = 0
        self.shadow = {k: (v.detach().clone().float()
                           if v.dtype.is_floating_point else v.detach().clone())
                       for k, v in model.state_dict().items()}
        self._backup = None

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        d = self.decay * (1.0 - math.exp(-self.num_updates / self.warmup_steps))
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach().float(), alpha=1.0 - d)
            else:
                s.copy_(v)

    def apply_to(self, model):
        """Swap EMA weights into the model (keep a backup for restore())."""
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict({k: v.to(dtype=self._backup[k].dtype)
                               for k, v in self.shadow.items()})

    def restore(self, model):
        """Swap the raw training weights back after evaluation."""
        if self._backup is not None:
            model.load_state_dict(self._backup)
            self._backup = None

    def state_dict(self):
        return self.shadow


def rotate_csi_features(feat, theta):
    """Apply h -> e^{j theta} h directly on the engineered CSI tensor
    (CSI_FEAT_CH channels: 12 for a ULA, 14 for a UPA).

    Exact identity, verified numerically by the self-check in:
      * rotate by theta : (Re, Im) = ch(0,1)
                          (cos phi, sin phi) = ch(3,4)
                          (cos angle A, sin angle A) = ch(6,7)
      * invariant       : log|h| (ch 2), log|A| (ch 5), and every
                          differential-phase pair -- ch(8,9) always, plus
                          ch(10,11) [/ ch(12,13) for a UPA] -- the global
                          phase cancels inside h[.,k+1] conj(h[.,k]).
    Geometry-agnostic by construction: only the 3 rotating pairs are ever
    touched, so this needs no change when CSI_FEAT_CH is 12 vs 14.
    """
    out = feat.clone()
    c, s = math.cos(theta), math.sin(theta)
    for (a, b) in ((0, 1), (3, 4), (6, 7)):
        xa, xb = feat[:, a], feat[:, b]
        out[:, a] = xa * c - xb * s
        out[:, b] = xa * s + xb * c
    return out


def denormalize_gps(pred_norm: torch.Tensor, bs_pos_raw: torch.Tensor,
                    clip: bool = True) -> torch.Tensor:
    """Denormalize: (B, 3) isotropically-scaled offset -> raw UE metres:

        UE = BS + (pred * OFF_SCALE + OFF_MEAN)

    If clip=True, predictions are clamped to a slightly-expanded physical
    GPS bounding box of the dataset (10% margin past min/max), keeping
    mean/RMSE metrics robust against an occasional far-out prediction from
    an unbounded regression head, without distorting median/P90.
    """
    o_mean = torch.as_tensor(OFF_MEAN, device=pred_norm.device, dtype=pred_norm.dtype)
    out = bs_pos_raw.to(pred_norm.dtype) + pred_norm * float(OFF_SCALE) + o_mean
    if clip:
        lo = torch.as_tensor(GPS_MIN, device=out.device, dtype=out.dtype)
        hi = torch.as_tensor(GPS_MAX, device=out.device, dtype=out.dtype)
        margin = 0.10 * (hi - lo)
        out = torch.clamp(out, min=lo - margin, max=hi + margin)
    return out


def build_param_groups(model, lr, weight_decay, attn_lr_mult=0.35):
    """Four AdamW groups.

    Weight decay applies only to >= 2-D weight matrices -- decaying every
    parameter uniformly, including pos_emb, fusion_tokens, seeds, cam_id,
    LayerScale and LayerNorm gains/biases, pulls the learned positional and
    latent embeddings toward zero and erodes exactly the positional
    information the architecture depends on.

    Attention parameters additionally train at attn_lr_mult * lr: a
    multi-layer pre-norm transformer isn't stable at the LR a CoordConv CNN
    prefers.
    """
    attn_markers = ('layers.', 'fusion_tokens', 'film.', 'refine_proj',
                    'refine_norm', 'refine_scale', '.pma.', 'seeds', 'to_tok')
    nodecay_markers = ('pos_emb', 'cam_id', 'fusion_tokens', 'seeds',
                       'log_beta', 'ls_cross', 'ls_self', 'ls_ffn',
                       'refine_scale', 'missing_emb')

    groups = {'base_decay': [], 'base_nodecay': [], 'attn_decay': [], 'attn_nodecay': []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_attn = any(k in n for k in attn_markers)
        no_decay = (p.ndim <= 1) or any(k in n for k in nodecay_markers)
        key = ('attn_' if is_attn else 'base_') + ('nodecay' if no_decay else 'decay')
        groups[key].append(p)

    out = [
        {'params': groups['base_decay'],   'lr': lr,                'weight_decay': weight_decay},
        {'params': groups['base_nodecay'], 'lr': lr,                'weight_decay': 0.0},
        {'params': groups['attn_decay'],   'lr': lr * attn_lr_mult, 'weight_decay': weight_decay},
        {'params': groups['attn_nodecay'], 'lr': lr * attn_lr_mult, 'weight_decay': 0.0},
    ]
    counts = {k: sum(p.numel() for p in v) for k, v in groups.items()}
    return [g for g in out if len(g['params']) > 0], counts


def _resolve_aux_loss_targets(model, aux_loss_modalities):
    """Validates AUX_LOSS_MODALITIES against the actual model once, before
    training starts, and returns {modality: force_drop_set} for only the
    entries usable on this checkpoint's config -- skipping (with a clear
    one-time printed reason) anything that doesn't apply, rather than
    crashing mid-epoch or silently doing nothing. force_drop_set for
    modality m is "every other enabled modality", so isolating m means
    force-dropping the rest -- reusing the same substitution mechanism
    GPSLocalizer.forward() implements for TEST_TIME_ABLATE."""
    resolved = {}
    if not aux_loss_modalities:
        return resolved
    if len(model.mod_names) <= 1:
        print(f"[--] AUX_LOSS_MODALITIES={aux_loss_modalities} has no effect: "
              f"this checkpoint has only one modality enabled "
              f"({model.mod_names}), so there is no OTHER modality to "
              f"force-drop for a solo-modality auxiliary pass.")
        return resolved
    for m in aux_loss_modalities:
        if m not in model.mod_names:
            print(f"[--] AUX_LOSS_MODALITIES: '{m}' skipped -- not an enabled "
                  f"modality in this run ({model.mod_names}).")
            continue
        others = set(model.mod_names) - {m}
        missing_others = others - set(model.missing_emb.keys())
        if missing_others:
            print(f"[--] AUX_LOSS_MODALITIES: '{m}' skipped -- isolating it "
                  f"requires force-dropping {sorted(others)}, but "
                  f"{sorted(missing_others)} has no missing_emb (trained with "
                  f"MOD_DROP_P==0 for that modality). Set MOD_DROP_P > 0 for "
                  f"{sorted(missing_others)} first.")
            continue
        resolved[m] = others
        print(f"[OK] auxiliary '{m}'-only loss enabled every training batch "
              f"(force-dropping {sorted(others)}), weight={AUX_LOSS_WEIGHT}")
    return resolved
def train_one_epoch(model, loader, optimizer, scaler, criterion, device, ema=None,
                    aux_targets=None, aux_weight=0.0):
    """aux_targets: {modality: force_drop_set} from _resolve_aux_loss_targets(),
    or None/empty to disable. When set, an additional deterministic
    solo-modality forward pass runs every batch for each target modality
    (disable_random_dropout=True, so it reliably isolates exactly that
    modality -- see GPSLocalizer.forward()), and its loss is added to the
    main fused-prediction loss, weighted by aux_weight. The physics-anchor
    loss term is excluded from the auxiliary loss (anchor_pred forced to
    None before calling criterion): it's a CSI-specific beamspace
    prediction and isn't meaningful when CSI may be one of the modalities
    forced absent for this pass.

    Returns (main_loss, aux_loss_or_None) -- main_loss is the original
    fused-prediction loss only, so existing history/plots/comparisons stay
    valid; the auxiliary contribution is tracked and returned separately."""
    model.train()
    total_loss, total_aux_loss, n = 0.0, 0.0, 0
    pbar = tqdm(loader, desc="  Train", leave=False)
    for targets, features, _ in pbar:
        targets  = targets.to(device, non_blocking=True)
        features = {k: v.to(device, non_blocking=True) for k, v in features.items()}
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=torch.cuda.is_available(), dtype=AMP_DTYPE):
            pred, anchor_pred, bin_logits, _ = model(features)
            loss = criterion(pred, anchor_pred, bin_logits, targets)

            aux_loss_sum = None
            if aux_targets and aux_weight > 0:
                for _m, _force_drop in aux_targets.items():
                    aux_pred, _aux_anchor, aux_bin_logits, _ = model(
                        features, force_drop=_force_drop,
                        disable_random_dropout=True)
                    _aux_l = criterion(aux_pred, None, aux_bin_logits, targets)
                    aux_loss_sum = _aux_l if aux_loss_sum is None else aux_loss_sum + _aux_l
            combined_loss = loss if aux_loss_sum is None else loss + aux_weight * aux_loss_sum

        if scaler is not None and scaler.is_enabled():
            scaler.scale(combined_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
        if ema is not None:
            ema.update(model)
        bs = targets.size(0)
        total_loss += loss.item() * bs
        if aux_loss_sum is not None:
            total_aux_loss += float(aux_loss_sum.item()) * bs
        n += bs
        postfix = {'loss': f"{total_loss / n:.6f}"}
        if aux_targets:
            postfix['aux'] = f"{total_aux_loss / n:.6f}"
        pbar.set_postfix(**postfix)
    return (total_loss / max(n, 1),
           (total_aux_loss / max(n, 1)) if aux_targets else None)


@torch.no_grad()
def evaluate_loss_and_error(model, loader, criterion, device):
    """Returns (mean_loss_in_scaled_space, mean_distance_error_in_metres,
    gate_means), where gate_means is {modality: mean reliability-gate value
    across all fusion layers and all validation samples this epoch} -- empty
    if USE_ATTN_REFINE is off. This is the direct readout of whether the
    per-sample gating mechanism is actually learning to discriminate
    between modalities, tracked every epoch. Single clean forward pass (no
    TTA), used for fast per-epoch validation and early stopping, so
    test-time TTA can never leak into model selection."""
    model.eval()
    total_loss, total_err, n = 0.0, 0.0, 0
    gate_sum, gate_n = {}, {}
    for targets, features, ue_raw in loader:
        targets  = targets.to(device, non_blocking=True)
        features = {k: v.to(device, non_blocking=True) for k, v in features.items()}
        ue_raw   = ue_raw.to(device, non_blocking=True)
        pred, anchor_pred, bin_logits, gate_diag = model(features)
        loss = criterion(pred, anchor_pred, bin_logits, targets)
        pred_raw = denormalize_gps(pred.float(), features['bs_pos_raw'])
        err = torch.norm(pred_raw - ue_raw, dim=1)    # 3-D Euclidean distance
        bs = targets.size(0)
        total_loss += loss.item() * bs
        total_err  += err.sum().item()
        n += bs
        for m, r_list in gate_diag.items():
            gate_sum[m] = gate_sum.get(m, 0.0) + sum(r.sum().item() for r in r_list)
            gate_n[m]   = gate_n.get(m, 0)     + sum(r.numel()      for r in r_list)
    gate_means = {m: gate_sum[m] / gate_n[m] for m in gate_sum if gate_n.get(m)}
    return total_loss / max(n, 1), total_err / max(n, 1), gate_means


@torch.no_grad()
def evaluate_full(model, loader, device, tta_k: int = 1, force_drop=None):
    """Returns per-sample predicted and ground-truth GPS (raw metres) plus
    the scaled-offset predictions, so every metric can be computed and
    saved to .mat.

    If tta_k > 1 and CSI is an input, the prediction is averaged over tta_k
    global phase rotations of the CSI -- a label-invariant nuisance
    transform applied exactly in feature space. Pure
    variance reduction, no feature recomputation, zero leakage.

    force_drop: forwarded to model(features, force_drop=...)
    unconditionally on every sample -- see GPSLocalizer.forward(). Used for
    test-time modality ablation (TEST_TIME_ABLATE), never during
    normal evaluation.

    This function does not alter tta_k based on force_drop. An earlier
    version auto-forced tta_k=1 whenever 'csi' was in force_drop, reasoning
    that CSI is about to be discarded so rotating it first wastes compute --
    that broke the moment force_drop turned out to be a no-op
    (GPSLocalizer.forward()'s own len(dense)>1 guard blocks substitution
    when the ablated modality is the only one enabled): CSI was never
    actually removed, but TTA got turned off anyway, producing a misleading
    "ablated" number that was really just "full_fusion minus TTA
    averaging". The caller now always controls tta_k explicitly; matching
    it between the full-fusion and ablated calls is what makes the
    comparison valid.
    """
    model.eval()
    all_pred_raw, all_gt_raw, all_pred_norm, all_gt_norm = [], [], [], []
    for targets, features, ue_raw in loader:
        targets  = targets.to(device, non_blocking=True)
        features = {k: v.to(device, non_blocking=True) for k, v in features.items()}
        ue_raw   = ue_raw.to(device, non_blocking=True)

        if tta_k > 1 and 'csi_feat' in features:
            preds = []
            base = features['csi_feat']
            for k in range(tta_k):
                fk = dict(features)
                fk['csi_feat'] = rotate_csi_features(base, 2.0 * math.pi * k / tta_k)
                p, _, _, _ = model(fk, force_drop=force_drop)
                preds.append(p.float())
            pred = torch.stack(preds, dim=0).mean(dim=0)
        else:
            pred, _, _, _ = model(features, force_drop=force_drop)
            pred = pred.float()

        pred_raw = denormalize_gps(pred, features['bs_pos_raw'])
        all_pred_norm.append(pred.cpu().numpy())
        all_gt_norm.append(targets.cpu().numpy())
        all_pred_raw.append(pred_raw.cpu().numpy())
        all_gt_raw.append(ue_raw.cpu().numpy())
    return (np.concatenate(all_pred_raw,  axis=0),
            np.concatenate(all_gt_raw,    axis=0),
            np.concatenate(all_pred_norm, axis=0),
            np.concatenate(all_gt_norm,   axis=0))


print("[OK] Scheduler + DFR loss + bias-corrected EMA + phase TTA + "
      "param groups + train/eval utilities defined (v5)")


with h5py.File(HDF5_FILE, 'r') as f:
    num_samples = int(f.attrs['num_samples'])
    print(f"HDF5: {HDF5_FILE}")
    print(f"Samples in HDF5: {num_samples:,}")
    for k in f.keys():
        print(f"  {k}: {f[k].shape}")

print(f"Clean samples available: {len(CLEAN_INDICES):,} / {num_samples:,}")

train_idx, val_idx, test_idx = TRAIN_IDX, VAL_IDX, TEST_IDX
n_clean = len(CLEAN_INDICES)
print(f"\nSplit (scene-contiguous, Option A) -> "
      f"Train {len(train_idx):,} | Val {len(val_idx):,} | Test {len(test_idx):,}")

split_file = OUTPUT_DIR / "split_indices.json"
with open(split_file, 'w') as f:
    json.dump({'train': train_idx, 'val': val_idx, 'test': test_idx,
               'split_method': 'scene_contiguous_block_optionA',
               'train_ratio': TRAIN_RATIO, 'val_ratio': VAL_RATIO,
               'num_samples_in_hdf5': num_samples,
               'num_clean_samples':   n_clean}, f)
print(f"[OK] Split indices saved -> {split_file}")

ds_kw = dict(
    hdf5_path=HDF5_FILE, valid_samples_json=VALID_SAMPLES_FILE,
    img_size=IMG_SIZE, lidar_points=LIDAR_POINTS,
    num_cameras_per_bs=NUM_CAMERAS_PER_BS,
    use_csi=USE_CSI, use_bs_pos=USE_BS_POS,
    use_camera=USE_CAMERA, use_lidar=USE_LIDAR,
    use_all_cameras=USE_ALL_CAMERAS,
    csi_std=CSI_STD,                                   # per-scenario scale
    logmag_mean=LOGMAG_MEAN, logmag_std=LOGMAG_STD,    # log-domain stats
    fftlog_mean=FFTLOG_MEAN, fftlog_std=FFTLOG_STD,
    off_mean=OFF_MEAN, off_scale=OFF_SCALE,            # scalar target scale
    gps_mean=GPS_MEAN, gps_std=GPS_STD,
    is_upa=IS_UPA, ant_rows=ANT_ROWS, ant_cols=ANT_COLS,  # geometry switch
)
dl_kw = dict(num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
if NUM_WORKERS > 0:
    dl_kw['persistent_workers'] = True
    dl_kw['prefetch_factor'] = 4

def _seed_worker(worker_id):
    s = SEED + worker_id
    np.random.seed(s)
    random.seed(s)

g = torch.Generator()
g.manual_seed(SEED)

train_ds = MultimodalGPSDataset(indices=train_idx, augment=True,
                                aug_phase=AUG_PHASE, aug_noise_p=AUG_NOISE_P,
                                aug_snr_db=AUG_SNR_DB, **ds_kw)
val_ds   = MultimodalGPSDataset(indices=val_idx,   augment=False, **ds_kw)
test_ds  = MultimodalGPSDataset(indices=test_idx,  augment=False, **ds_kw)

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  drop_last=True,
                          worker_init_fn=_seed_worker, generator=g, **dl_kw)
val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, **dl_kw)
test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, **dl_kw)

print(f"Loaders -> Train {len(train_loader)} | Val {len(val_loader)} | Test {len(test_loader)}")

_probe_n = min(2048, len(train_ds))
_probe = np.stack([train_ds[i][0].numpy() for i in range(0, _probe_n,
                                                        max(1, _probe_n // 256))])
print(f"  Target check -> per-axis std {np.round(_probe.std(0), 4).tolist()} | "
      f"RMS norm {float(np.sqrt((_probe ** 2).sum(1).mean())):.4f} "
      f"(1 unit = {OFF_SCALE:.2f} m)")


if IS_LATE_FUSION:
    print("=" * 60)
    print("LATE FUSION (v16) -- loading frozen single-modality experts")
    print("=" * 60)
    _csi_expert = _load_frozen_base('csi', DEVICE)
    _cam_expert = _load_frozen_base('cam', DEVICE)
    model = LateFusionLocalizer(
        csi_model=_csi_expert, cam_model=_cam_expert,
        out_dim=OUT_DIM, num_bins=NUM_BINS,
        csi_embed_dim=CSI_EMBED_DIM, cam_embed_dim=CAM_EMBED_DIM,
        emb_hidden=LATE_EMB_HIDDEN,
        hidden=LATE_HIDDEN, dropout=LATE_DROPOUT,
        mod_drop_p=LATE_MOD_DROP_P, gate_clamp=LATE_GATE_CLAMP,
        corr_scale=LATE_CORR_SCALE,
    ).to(DEVICE)
    LAMBDA_ANCHOR_EFF   = 0.0
    AUX_LOSS_MODALITIES = []
else:
    LAMBDA_ANCHOR_EFF = LAMBDA_ANCHOR if USE_CSI else 0.0
    model = GPSLocalizer(
        use_csi=USE_CSI, use_bs_pos=USE_BS_POS,
        use_camera=USE_CAMERA, use_lidar=USE_LIDAR,
        use_all_cameras=USE_ALL_CAMERAS, num_cameras=NUM_CAMERAS,
        csi_embed_dim=CSI_EMBED_DIM, pos_embed_dim=POS_EMBED_DIM,
        cam_embed_dim=CAM_EMBED_DIM, lidar_embed_dim=LIDAR_EMBED_DIM,
        fusion_dim=FUSION_DIM, dropout=DROPOUT,
        csi_width=CSI_WIDTH, csi_blocks=CSI_BLOCKS,
        csi_flatten_ch=CSI_FLATTEN_CH, cam_reduce_ch=CAM_REDUCE_CH,
        num_ant=NUM_ANT, num_sub=NUM_SUB, csi_in_ch=CSI_FEAT_CH, out_dim=OUT_DIM,
        img_hw=IMG_SIZE,
        use_attn_refine=USE_ATTN_REFINE, attn_dim=ATTN_DIM, num_heads=NUM_HEADS,
        num_fusion_tokens=NUM_FUSION_TOKENS, num_fusion_layers=NUM_FUSION_LAYERS,
        num_lidar_tokens=NUM_LIDAR_TOKENS,
        use_dist_head=USE_DIST_HEAD, num_bins=NUM_BINS, bin_range=BIN_RANGE,
        axis_scale=AXIS_SCALE, mod_drop_p=MOD_DROP_P, ls_init=LS_INIT,
        use_csi_confidence_gate=USE_CSI_CONFIDENCE_GATE,
        use_cam_confidence_gate=USE_CAM_CONFIDENCE_GATE,
        is_upa=IS_UPA, ant_rows=ANT_ROWS, ant_cols=ANT_COLS,
    ).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = total_params - train_params
if IS_LATE_FUSION:
    csi_params = sum(p.numel() for p in model.csi_model.csi_encoder.parameters())
else:
    csi_params = (sum(p.numel() for p in model.csi_encoder.parameters())
                  if USE_CSI else 0)

print("=" * 60)
print("MODEL")
print("=" * 60)
print(f"  Architecture  : {ARCH_TAG} (dense readout + zero-init attn residual)")
print(f"  Parameters    : {train_params:,}")
print(f"  CSI           : {USE_CSI}  ({CSI_FEAT_CH}-ch features, encoder {csi_params:,} params)")
print(f"  BS-pos        : {USE_BS_POS}")
print(f"  Camera        : {USE_CAMERA}" + (f" ({NUM_CAMERAS} cam)" if USE_CAMERA else ""))
print(f"  LiDAR         : {USE_LIDAR}" + (f" ({NUM_LIDAR_TOKENS} PMA tokens)" if USE_LIDAR else ""))
print(f"  Dense trunk   : {FUSION_DIM}")
print(f"  Attn refine   : {USE_ATTN_REFINE}"
      + (f" ({NUM_FUSION_TOKENS} tokens x {NUM_FUSION_LAYERS} layers, dim {ATTN_DIM})"
         if USE_ATTN_REFINE else ""))
print(f"  Dist head     : {USE_DIST_HEAD}" + (f" ({NUM_BINS} bins/axis)" if USE_DIST_HEAD else ""))
print(f"  FUSION MODE   : {FUSION_MODE}"
      + ("  (two-stage, frozen experts + CADPool)" if IS_LATE_FUSION
         else "  (joint end-to-end feature fusion)"))
if IS_LATE_FUSION:
    _pool_params = sum(p.numel() for p in model.pool.parameters())
    print(f"  Frozen experts: {frozen_params:,} params (not trained)")
    print(f"  CADPool       : {_pool_params:,} trainable params "
          f"({CADPool.N_DESC} descriptors/axis x {OUT_DIM} axes -> "
          f"{LATE_HIDDEN} hidden -> {3 * OUT_DIM} out)")
    print(f"                  zero-init => epoch 0 == precision-weighted log-pool")
    print(f"                  mod-drop {LATE_MOD_DROP_P} | gate clamp +-{LATE_GATE_CLAMP} "
          f"| corr scale {LATE_CORR_SCALE}")
print("=" * 60)

_bin_centers = model.bin_centers.detach().cpu() if USE_DIST_HEAD else None
criterion = DFRLoss(beta=LOSS_BETA_START, lambda_dist=LAMBDA_DIST,
                    lambda_dfl=LAMBDA_DFL if USE_DIST_HEAD else 0.0,
                    lambda_anchor=LAMBDA_ANCHOR_EFF,
                    bin_centers=_bin_centers)

if RUN_MODE == 'train_and_eval':
    param_groups, group_counts = build_param_groups(
        model, LR, WEIGHT_DECAY, attn_lr_mult=ATTN_LR_MULT)
    optimizer = optim.AdamW(param_groups, lr=LR, weight_decay=WEIGHT_DECAY,
                            betas=(0.9, 0.99), eps=1e-8)
    scheduler = get_cosine_warmup_scheduler(optimizer, WARMUP_EPOCHS, EPOCHS, MIN_LR, LR)
    scaler    = GradScaler(enabled=USE_GRAD_SCALER)
    ema       = ModelEMA(model, decay=EMA_DECAY,
                         warmup_steps=max(2000, len(train_loader) * 3))
    print(f"Optimizer : AdamW (lr={LR}, wd={WEIGHT_DECAY}, betas=(0.9, 0.99))")
    print(f"  groups  : base decay {group_counts['base_decay']:,} | "
          f"base no-decay {group_counts['base_nodecay']:,} | "
          f"attn decay {group_counts['attn_decay']:,} (lr x{ATTN_LR_MULT}) | "
          f"attn no-decay {group_counts['attn_nodecay']:,}")
    print(f"Scheduler : CosineWarmup (warmup={WARMUP_EPOCHS}, min_lr={MIN_LR})")
    print(f"AMP       : {AMP_DTYPE}  |  Grad-clip: {GRAD_CLIP}  |  EMA decay: {EMA_DECAY}")
    print(f"Loss      : SmoothL1(beta {LOSS_BETA_START} -> {LOSS_BETA_END} annealed) "
          f"+ {LAMBDA_DIST}*mean||.||2")
    print(f"            + {LAMBDA_DFL}*DFL + {LAMBDA_ANCHOR}*anchor, "
          f"on ISOTROPICALLY-scaled OFFSET (1 unit = {OFF_SCALE:.2f} m)")
else:
    print(f"RUN_MODE = 'eval_only' -- skipping optimizer/scheduler setup")


def _make_dummy_features(device):
    feats = {}
    if USE_CSI:
        feats['csi_feat']   = torch.zeros(1, CSI_FEAT_CH, NUM_ANT, NUM_SUB, device=device)
    feats['bs_pos_raw'] = torch.zeros(1, 3, device=device)
    if USE_BS_POS:
        feats['bs_pos'] = torch.zeros(1, 3, device=device)
    if USE_CAMERA:
        n_cams = NUM_CAMERAS if USE_ALL_CAMERAS else 1
        feats['images'] = torch.zeros(1, n_cams, 3, IMG_H, IMG_W, device=device)
    if USE_LIDAR:
        feats['lidar'] = torch.zeros(1, LIDAR_POINTS, 3, device=device)
    return feats


model_flops = None
try:
    from fvcore.nn import FlopCountAnalysis
    model.eval()
    with torch.no_grad():
        dummy = _make_dummy_features(DEVICE)
        fca = FlopCountAnalysis(model, (dummy,))
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        model_flops = int(fca.total())
    print(f"[OK] Model FLOPs (fvcore): {model_flops:,}   (~{model_flops/1e6:.2f} MFLOPs / sample)")
except Exception as e:
    print(f"[!]  Could not compute FLOPs via fvcore ({e}). Reporting params only.")

print(f"  Total params      : {total_params:,}")
print(f"  Trainable params  : {train_params:,}")

if USE_CSI:
    _csi_enc = model.csi_model.csi_encoder if IS_LATE_FUSION else model.csi_encoder
    dense_readout = sum(p.numel() for p in _csi_enc.dense.parameters())
    token_readout = (sum(p.numel() for p in _csi_enc.proj.parameters())
                     + _csi_enc.pos_emb.numel())
    _frozen_note = "  [FROZEN]" if IS_LATE_FUSION else "  (v4 had none -- this is the fix)"
    print(f"  CSI dense readout : {dense_readout:,}{_frozen_note}")
    print(f"  CSI token readout : {token_readout:,}")


best_model_path  = OUTPUT_DIR / "best_model.pth"
final_model_path = OUTPUT_DIR / "final_model.pth"
history_json     = OUTPUT_DIR / "training_history.json"
history_csv      = OUTPUT_DIR / "training_history.csv"


def _checkpoint_payload(epoch, val_loss, val_err):
    return {
        'epoch': epoch,
        'val_loss': val_loss,
        'val_mean_error_m': val_err,
        'arch_tag': ARCH_TAG,
        'gps_mean': GPS_MEAN.tolist(), 'gps_std': GPS_STD.tolist(),
        'gps_min':  GPS_MIN.tolist(),  'gps_max': GPS_MAX.tolist(),
        'off_mean': OFF_MEAN.tolist(), 'off_std': OFF_STD.tolist(),
        'off_scale': float(OFF_SCALE), 'axis_scale': AXIS_SCALE.tolist(),
        'csi_std':  float(CSI_STD), 'csi_feat_ch': int(CSI_FEAT_CH),
        'logmag_mean': float(LOGMAG_MEAN), 'logmag_std': float(LOGMAG_STD),
        'fftlog_mean': float(FFTLOG_MEAN), 'fftlog_std': float(FFTLOG_STD),
        'csi_embed_dim': CSI_EMBED_DIM, 'pos_embed_dim': POS_EMBED_DIM,
        'cam_embed_dim': CAM_EMBED_DIM, 'lidar_embed_dim': LIDAR_EMBED_DIM,
        'fusion_dim': FUSION_DIM, 'csi_width': CSI_WIDTH, 'csi_blocks': CSI_BLOCKS,
        'use_attn_refine': USE_ATTN_REFINE, 'attn_dim': ATTN_DIM,
        'num_heads': NUM_HEADS, 'num_fusion_tokens': NUM_FUSION_TOKENS,
        'num_fusion_layers': NUM_FUSION_LAYERS, 'num_lidar_tokens': NUM_LIDAR_TOKENS,
        'use_dist_head': USE_DIST_HEAD, 'num_bins': NUM_BINS, 'bin_range': BIN_RANGE,
        'lambda_dist': LAMBDA_DIST, 'lambda_dfl': LAMBDA_DFL,
        'lambda_anchor': LAMBDA_ANCHOR_EFF, 'mod_drop_p': MOD_DROP_P,
        'fusion_mode': FUSION_MODE, 'is_late_fusion': bool(IS_LATE_FUSION),
        'num_ant': int(NUM_ANT), 'num_sub': int(NUM_SUB),
        'ant_shape': [int(ANT_ROWS), int(ANT_COLS)], 'out_dim': int(OUT_DIM),
    }


if RUN_MODE == 'train_and_eval':
    print("\n" + "=" * 72)
    print("TRAINING")
    print("=" * 72)
    print(f"  Experiment : {EXPERIMENT_NAME}")
    print(f"  Output     : {OUTPUT_DIR}")
    print(f"  Patience   : {PATIENCE}")
    print(f"  Target     : 1 loss unit = {OFF_SCALE:.2f} m")
    print("=" * 72 + "\n")

    history = {
        'train_loss': [], 'val_loss': [],
        'val_mean_error_m': [],            # in metres (EMA weights)
        'lr': [], 'beta': [],
        'refine_scale_absmean': [],
        'pool_gain_absmean': [],
        'gate_mean': {m: [] for m in
                     (['csi'] if USE_CSI else []) +
                     (['cam'] if USE_CAMERA else []) +
                     (['lidar'] if USE_LIDAR else []) +
                     ((['csi_confidence', 'cam_confidence']
                       if (USE_CSI and USE_CAMERA and not IS_LATE_FUSION) else [])
                      + (['conflict', 'prec_csi'] if IS_LATE_FUSION else []))},
        'aux_loss': [],
    }
    _aux_targets = _resolve_aux_loss_targets(model, AUX_LOSS_MODALITIES)

    best_val_err      = float('inf')
    best_epoch        = 0
    epochs_no_improve = 0
    t_start           = time.time()
    set_seed(42)
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        beta_now = beta_at_epoch(epoch, EPOCHS, LOSS_BETA_START, LOSS_BETA_END)
        criterion.set_beta(beta_now)

        train_loss, aux_loss_val = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, DEVICE, ema=ema,
            aux_targets=_aux_targets, aux_weight=AUX_LOSS_WEIGHT)

        ema.apply_to(model)
        val_loss, val_err, gate_means = evaluate_loss_and_error(
            model, val_loader, criterion, DEVICE)

        lr_now = optimizer.param_groups[0]['lr']
        dt = time.time() - t0

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mean_error_m'].append(val_err)
        history['lr'].append(lr_now)
        history['beta'].append(beta_now)
        history['aux_loss'].append(aux_loss_val if aux_loss_val is not None else float('nan'))
        if USE_ATTN_REFINE and not IS_LATE_FUSION:
            history['refine_scale_absmean'].append(
                float(model.refine_scale.detach().abs().mean()))
        if IS_LATE_FUSION:
            history['pool_gain_absmean'].append(
                float(model.pool.net[-1].weight.detach().abs().mean()))
        for m in history['gate_mean']:
            history['gate_mean'][m].append(gate_means.get(m, float('nan')))

        tag = ""
        if val_err < best_val_err:
            best_val_err      = val_err
            best_epoch        = epoch
            epochs_no_improve = 0
            ckpt = _checkpoint_payload(epoch, val_loss, val_err)
            ckpt['model_state_dict']     = model.state_dict()      # EMA weights
            ckpt['optimizer_state_dict'] = optimizer.state_dict()
            torch.save(ckpt, best_model_path)
            tag = f"  [best {val_err:.3f} m]"
        else:
            epochs_no_improve += 1

        ema.restore(model)
        scheduler.step()

        print(f"Ep {epoch:3d}/{EPOCHS}  Tr {train_loss:.5f}  Val {val_loss:.5f}  "
              f"MeanErr {val_err:.3f} m  LR {lr_now:.2e}  b {beta_now:.3f}  "
              f"{dt:.0f}s{tag}"
              + (f"  aux {aux_loss_val:.5f}" if aux_loss_val is not None else "")
              + (f"  |gamma| {history['refine_scale_absmean'][-1]:.4f}"
                 if USE_ATTN_REFINE and not IS_LATE_FUSION and history['refine_scale_absmean']
                 else "")
              + (f"  |pool_g| {history['pool_gain_absmean'][-1]:.4f}"
                 if IS_LATE_FUSION and history['pool_gain_absmean'] else "")
              + (f"  gates[{', '.join(f'{m}={history['gate_mean'][m][-1]:.2f}' for m in history['gate_mean'])}]"
                 if history['gate_mean'] else ""))

        if epochs_no_improve >= PATIENCE:
            print(f"\n[..] Early stop at epoch {epoch} (patience={PATIENCE})")
            break

    total_time = time.time() - t_start
    raw_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ema.apply_to(model)
    ckpt = _checkpoint_payload(epoch, val_loss, val_err)
    ckpt['model_state_dict']     = model.state_dict()              # EMA weights
    ckpt['raw_model_state_dict'] = raw_state                       # last SGD iterate
    torch.save(ckpt, final_model_path)
    ema.restore(model)

    if USE_ATTN_REFINE and not IS_LATE_FUSION:
        rs = model.refine_scale.detach().abs()
        print(f"\n  Attention LayerScale |gamma|: mean {rs.mean():.4f}  "
              f"max {rs.max():.4f}   (0.0 == model stayed at the dense trunk)")
        rs_trend = history.get('refine_scale_absmean', [])
        if len(rs_trend) >= 8:
            q = max(1, len(rs_trend) // 4)
            early, late = np.mean(rs_trend[:q]), np.mean(rs_trend[-q:])
            direction = ("still rising -- consider more epochs" if late > early * 1.15
                        else "plateaued" if late < early * 1.15 and late > early * 0.85
                        else "fell")
            print(f"  |gamma| trend: first-{q}-epoch mean {early:.4f} -> "
                  f"last-{q}-epoch mean {late:.4f}  ({direction})")
    elif IS_LATE_FUSION:
        pg = model.pool.net[-1].weight.detach().abs()
        print(f"\n  CADPool last-layer |weight|: mean {pg.mean():.4f}  "
              f"max {pg.max():.4f}   (0.0 == pool stayed at the zero-init "
              f"precision-weighted classical estimator)")
        pg_trend = history.get('pool_gain_absmean', [])
        if len(pg_trend) >= 8:
            q = max(1, len(pg_trend) // 4)
            early, late = np.mean(pg_trend[:q]), np.mean(pg_trend[-q:])
            direction = ("still rising -- consider more epochs" if late > early * 1.15
                        else "plateaued" if late < early * 1.15 and late > early * 0.85
                        else "fell")
            print(f"  |weight| trend: first-{q}-epoch mean {early:.4f} -> "
                  f"last-{q}-epoch mean {late:.4f}  ({direction})")

    for m in history.get('gate_mean', {}):
        gm = history['gate_mean'][m]
        if gm:
            print(f"  gate[{m}] mean over training: {np.mean(gm):.3f}  "
                  f"(final epoch: {gm[-1]:.3f}; 0.5 == no preference, "
                  f"far from 0.5 == branch has an opinion about this modality)")

    print("\n" + "=" * 72)
    print(f"Training done -- best val mean error: {best_val_err:.3f} m  (ep {best_epoch})")
    print(f"Time: {total_time / 60:.1f} min  |  Epochs: {epoch}")
    print("=" * 72)
else:
    print("RUN_MODE = 'eval_only' -- skipping training.")


if RUN_MODE == 'train_and_eval':
    with open(history_json, 'w') as f:
        json.dump(history, f, indent=2)
    with open(history_csv, 'w') as f:
        _gate_mods = [m for m in history.get('gate_mean', {}) if history['gate_mean'][m]]
        _has_refine = len(history.get('refine_scale_absmean', [])) > 0
        _has_pool_g = len(history.get('pool_gain_absmean', [])) > 0
        _cols = "epoch,train_loss,val_loss,val_mean_error_m,lr,beta"
        if _has_refine:
            _cols += ",refine_scale_absmean"
        if _has_pool_g:
            _cols += ",pool_gain_absmean"
        _cols += "".join(f",gate_{m}" for m in _gate_mods)
        f.write(_cols + "\n")
        for i in range(len(history['train_loss'])):
            row = (f"{i+1},{history['train_loss'][i]:.8f},{history['val_loss'][i]:.8f},"
                  f"{history['val_mean_error_m'][i]:.6f},{history['lr'][i]:.8e},"
                  f"{history['beta'][i]:.6f}")
            if _has_refine:
                rs_i = history['refine_scale_absmean'][i] if i < len(history['refine_scale_absmean']) else float('nan')
                row += f",{rs_i:.6f}"
            if _has_pool_g:
                pg_i = history['pool_gain_absmean'][i] if i < len(history['pool_gain_absmean']) else float('nan')
                row += f",{pg_i:.6f}"
            for m in _gate_mods:
                gv = history['gate_mean'][m][i] if i < len(history['gate_mean'][m]) else float('nan')
                row += f",{gv:.6f}"
            f.write(row + "\n")
    print(f"[OK] History saved -> {history_json}  and  {history_csv}")
else:
    print("Skipped (eval_only mode).")


print("\n" + "=" * 72)
print("TEST EVALUATION")
print("=" * 72)

if not best_model_path.exists():
    raise FileNotFoundError(
        f"No best_model.pth found at {best_model_path}.\n"
        f"Run with RUN_MODE='train_and_eval' first."
    )

try:
    ckpt = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
except TypeError:
    ckpt = torch.load(best_model_path, map_location=DEVICE)
model.load_state_dict(ckpt['model_state_dict'])     
print(f"Loaded best model (epoch {ckpt['epoch']}, arch={ckpt.get('arch_tag', 'v1')})")

_ck_scale = ckpt.get('off_scale', None)
if _ck_scale is not None and abs(float(_ck_scale) - float(OFF_SCALE)) > 1e-6:
    raise RuntimeError(
        f"Checkpoint OFF_SCALE ({_ck_scale}) != current OFF_SCALE ({OFF_SCALE}). "
        f"This checkpoint was trained with a different target normalization."
    )

tta_k = TTA_PHASE_K if USE_CSI else 1
if tta_k > 1:
    print(f"TTA: averaging over {tta_k} CSI phase rotations (exact, feature-space)")

test_pred_raw, test_gt_raw, test_pred_norm, test_gt_norm = evaluate_full(
    model, test_loader, DEVICE, tta_k=tta_k)

diffs_3d   = test_pred_raw - test_gt_raw           # (N, 3)
err_3d     = np.linalg.norm(diffs_3d, axis=1)      # (N,)
err_2d     = np.linalg.norm(diffs_3d[:, :2], axis=1)
err_per_axis_mae = np.mean(np.abs(diffs_3d), axis=0)

mean_error_3d   = float(err_3d.mean())
median_error_3d = float(np.median(err_3d))
p90_error_3d    = float(np.percentile(err_3d, 90))
p95_error_3d    = float(np.percentile(err_3d, 95))
rmse_3d         = float(np.sqrt((err_3d ** 2).mean()))

mean_error_2d   = float(err_2d.mean())
median_error_2d = float(np.median(err_2d))
p90_error_2d    = float(np.percentile(err_2d, 90))
p95_error_2d    = float(np.percentile(err_2d, 95))
rmse_2d         = float(np.sqrt((err_2d ** 2).mean()))

acc_1m  = float((err_3d <= 1.0).mean() * 100.0)
acc_2m  = float((err_3d <= 2.0).mean() * 100.0)
acc_5m  = float((err_3d <= 5.0).mean() * 100.0)

print(f"\n  ----  3D positioning error (metres)  ----")
print(f"  Mean     : {mean_error_3d:.3f}")
print(f"  Median   : {median_error_3d:.3f}")
print(f"  RMSE     : {rmse_3d:.3f}")
print(f"  P90      : {p90_error_3d:.3f}")
print(f"  P95      : {p95_error_3d:.3f}")
print(f"  ----  2D positioning error (metres)  ----")
print(f"  Mean     : {mean_error_2d:.3f}")
print(f"  Median   : {median_error_2d:.3f}")
print(f"  RMSE     : {rmse_2d:.3f}")
print(f"  P90      : {p90_error_2d:.3f}")
print(f"  P95      : {p95_error_2d:.3f}")
print(f"  ----  Per-axis MAE (metres)  ----")
print(f"  X / Y / Z : {err_per_axis_mae[0]:.3f} / {err_per_axis_mae[1]:.3f} / {err_per_axis_mae[2]:.3f}")
print(f"  ----  Accuracy within threshold (3D)  ----")
print(f"  <= 1 m : {acc_1m:.2f} %   <= 2 m : {acc_2m:.2f} %   <= 5 m : {acc_5m:.2f} %")
print("=" * 72)


ablation_result = None
if TEST_TIME_ABLATE is not None:
    _abl_enabled = {'csi': USE_CSI, 'cam': USE_CAMERA, 'lidar': USE_LIDAR}
    if not _abl_enabled.get(TEST_TIME_ABLATE, False):
        print(f"\n[--] TEST_TIME_ABLATE='{TEST_TIME_ABLATE}' skipped: that modality "
              f"is not enabled in this run (USE_{TEST_TIME_ABLATE.upper()}=False), "
              f"so there is nothing to ablate.")
    elif TEST_TIME_ABLATE not in model.missing_emb:
        print(f"\n[--] TEST_TIME_ABLATE='{TEST_TIME_ABLATE}' skipped: this checkpoint "
              f"has no missing_emb['{TEST_TIME_ABLATE}'] -- it was trained with "
              f"MOD_DROP_P['{TEST_TIME_ABLATE}']==0, so there is no learned "
              f"'this sensor is absent' representation to substitute. Retrain with "
              f"MOD_DROP_P['{TEST_TIME_ABLATE}'] > 0 first if you want this ablation.")
    elif len(model.mod_names) <= 1:
        print(f"\n[--] TEST_TIME_ABLATE='{TEST_TIME_ABLATE}' skipped: this checkpoint "
              f"has only ONE modality enabled ({model.mod_names}). "
              f"GPSLocalizer.forward() refuses to force-drop a model's sole input "
              f"(there is nothing left for the trunk to fall back on), so this "
              f"would not be a real ablation -- it would silently do nothing to "
              f"the prediction. Point TEST_TIME_ABLATE at a FUSION checkpoint "
              f"(multiple modalities enabled) to get a meaningful answer.")
    else:
        print("\n" + "=" * 72)
        print(f"TEST-TIME ABLATION: '{TEST_TIME_ABLATE}' forced absent "
              f"(same weights, same {len(test_ds)} test samples)")
        print("=" * 72)
        abl_tta_k = tta_k
        abl_pred_raw, abl_gt_raw, _, _ = evaluate_full(
            model, test_loader, DEVICE, tta_k=abl_tta_k,
            force_drop={TEST_TIME_ABLATE})
        abl_diffs = abl_pred_raw - abl_gt_raw
        abl_err   = np.linalg.norm(abl_diffs, axis=1)
        abl_mean, abl_median = float(abl_err.mean()), float(np.median(abl_err))
        abl_rmse  = float(np.sqrt((abl_err ** 2).mean()))
        abl_p90   = float(np.percentile(abl_err, 90))
        abl_p95   = float(np.percentile(abl_err, 95))

        def _delta(full, abl):
            sign = '+' if abl >= full else ''
            pct  = (abl - full) / full * 100.0 if full > 0 else float('nan')
            return f"{abl:.3f}  ({sign}{pct:.1f}% vs full fusion)"

        print(f"  {'metric':<10} {'full fusion':>12} {'-> ablated (' + TEST_TIME_ABLATE + ')':>28}")
        print(f"  {'mean':<10} {mean_error_3d:>12.3f} {'-> ' + _delta(mean_error_3d, abl_mean):>28}")
        print(f"  {'median':<10} {median_error_3d:>12.3f} {'-> ' + _delta(median_error_3d, abl_median):>28}")
        print(f"  {'rmse':<10} {rmse_3d:>12.3f} {'-> ' + _delta(rmse_3d, abl_rmse):>28}")
        print(f"  {'p90':<10} {p90_error_3d:>12.3f} {'-> ' + _delta(p90_error_3d, abl_p90):>28}")
        print(f"  {'p95':<10} {p95_error_3d:>12.3f} {'-> ' + _delta(p95_error_3d, abl_p95):>28}")
        if abl_mean <= mean_error_3d * 1.03:
            print(f"\n  -> Removing '{TEST_TIME_ABLATE}' barely changed mean error. The "
                  f"trained model does not appear to rely on it much.")
        else:
            print(f"\n  -> Removing '{TEST_TIME_ABLATE}' clearly hurt mean error. The "
                  f"trained model IS using it.")
        print("=" * 72)

        ablation_result = {
            'ablated_modality': TEST_TIME_ABLATE,
            'full_fusion': {'mean': mean_error_3d, 'median': median_error_3d,
                            'rmse': rmse_3d, 'p90': p90_error_3d, 'p95': p95_error_3d},
            'ablated': {'mean': abl_mean, 'median': abl_median,
                       'rmse': abl_rmse, 'p90': abl_p90, 'p95': abl_p95},
        }
        abl_path = OUTPUT_DIR / f"ablation_{TEST_TIME_ABLATE}.txt"
        with open(abl_path, 'w') as _af:
            _af.write(f"Test-time modality ablation: '{TEST_TIME_ABLATE}' forced absent\n")
            _af.write(f"Checkpoint: {best_model_path}\n")
            _af.write(f"Experiment: {EXPERIMENT_NAME}\n")
            _af.write(f"Test samples: {len(test_ds)}\n")
            _af.write(f"TTA phase rotations (SAME for both rows below): {abl_tta_k}\n")
            _af.write(f"Modalities enabled in this checkpoint: {model.mod_names}\n\n")
            _af.write(f"{'metric':<10}{'full_fusion':>14}{'ablated':>14}{'pct_change':>14}\n")
            for k, full_v, abl_v in [
                ('mean', mean_error_3d, abl_mean), ('median', median_error_3d, abl_median),
                ('rmse', rmse_3d, abl_rmse), ('p90', p90_error_3d, abl_p90),
                ('p95', p95_error_3d, abl_p95)]:
                pct = (abl_v - full_v) / full_v * 100.0 if full_v > 0 else float('nan')
                _af.write(f"{k:<10}{full_v:>14.4f}{abl_v:>14.4f}{pct:>+13.1f}%\n")
        print(f"[OK] saved -> {abl_path}")


if RUN_MODE == 'eval_only' and history_json.exists():
    with open(history_json) as f:
        history = json.load(f)
elif RUN_MODE == 'eval_only':
    history = {'train_loss': [], 'val_loss': [], 'val_mean_error_m': [], 'lr': [], 'beta': []}

mod_list = []
if USE_CSI:    mod_list.append("CSI")
if USE_BS_POS: mod_list.append("BS-pos")
if USE_CAMERA: mod_list.append(f"Camera({NUM_CAMERAS if USE_ALL_CAMERAS else 1})")
if USE_LIDAR:  mod_list.append("LiDAR")

mat_dict = {
    'test_pred_raw':  test_pred_raw.astype(np.float64),
    'test_gt_raw':    test_gt_raw.astype(np.float64),
    'test_pred_norm': test_pred_norm.astype(np.float64),
    'test_gt_norm':   test_gt_norm.astype(np.float64),
    'test_err_3d':    err_3d.astype(np.float64),
    'test_err_2d':    err_2d.astype(np.float64),
    'mean_error_3d':   mean_error_3d,
    'median_error_3d': median_error_3d,
    'rmse_3d':         rmse_3d,
    'p90_error_3d':    p90_error_3d,
    'p95_error_3d':    p95_error_3d,
    'mean_error_2d':   mean_error_2d,
    'median_error_2d': median_error_2d,
    'rmse_2d':         rmse_2d,
    'p90_error_2d':    p90_error_2d,
    'p95_error_2d':    p95_error_2d,
    'mae_per_axis':    err_per_axis_mae.astype(np.float64),
    'acc_within_1m':   acc_1m,
    'acc_within_2m':   acc_2m,
    'acc_within_5m':   acc_5m,

    'train_loss':       np.asarray(history.get('train_loss', []), dtype=np.float64),
    'val_loss':         np.asarray(history.get('val_loss',   []), dtype=np.float64),
    'val_mean_error_m': np.asarray(history.get('val_mean_error_m', []), dtype=np.float64),
    'lr_history':       np.asarray(history.get('lr', []), dtype=np.float64),
    'beta_history':     np.asarray(history.get('beta', []), dtype=np.float64),

    'num_params':       float(train_params),
    'num_total_params': float(total_params),
    'flops':            float(model_flops) if model_flops is not None else float('nan'),

    'gps_mean': GPS_MEAN.astype(np.float64),
    'gps_std':  GPS_STD.astype(np.float64),
    'gps_min':  GPS_MIN.astype(np.float64),
    'gps_max':  GPS_MAX.astype(np.float64),

    'off_mean':         OFF_MEAN.astype(np.float64),
    'off_std':          OFF_STD.astype(np.float64),
    'off_scale':        float(OFF_SCALE),          # scalar target scale
    'axis_scale':       AXIS_SCALE.astype(np.float64),
    'csi_std_scenario': float(CSI_STD),
    'csi_feat_ch':      int(CSI_FEAT_CH),
    'logmag_mean':      float(LOGMAG_MEAN),
    'logmag_std':       float(LOGMAG_STD),
    'fftlog_mean':      float(FFTLOG_MEAN),
    'fftlog_std':       float(FFTLOG_STD),

    'seed':            int(SEED),
    'num_train':       int(len(train_idx)),
    'num_val':         int(len(val_idx)),
    'num_test':        int(len(test_idx)),
    'num_samples':         int(num_samples),
    'num_clean_samples':   int(len(CLEAN_INDICES)),

    'batch_size':       int(BATCH_SIZE),
    'epochs':           int(EPOCHS),
    'lr':               float(LR),
    'weight_decay':     float(WEIGHT_DECAY),
    'patience':         int(PATIENCE),
    'warmup_epochs':    int(WARMUP_EPOCHS),
    'min_lr':           float(MIN_LR),
    'csi_embed_dim':     int(CSI_EMBED_DIM),
    'pos_embed_dim':     int(POS_EMBED_DIM),
    'cam_embed_dim':     int(CAM_EMBED_DIM),
    'lidar_embed_dim':   int(LIDAR_EMBED_DIM),
    'fusion_dim':        int(FUSION_DIM),
    'csi_width':         int(CSI_WIDTH),
    'csi_blocks':        int(CSI_BLOCKS),
    'use_attn_refine':   int(USE_ATTN_REFINE),
    'attn_dim':          int(ATTN_DIM),
    'num_heads':         int(NUM_HEADS),
    'num_fusion_tokens': int(NUM_FUSION_TOKENS),
    'num_fusion_layers': int(NUM_FUSION_LAYERS),
    'num_lidar_tokens':  int(NUM_LIDAR_TOKENS),
    'attn_lr_mult':      float(ATTN_LR_MULT),
    'use_dist_head':     int(USE_DIST_HEAD),
    'num_bins':          int(NUM_BINS),
    'bin_range':         float(BIN_RANGE),
    'dropout':          float(DROPOUT),
    'grad_clip':        float(GRAD_CLIP),
    'loss_beta_start':  float(LOSS_BETA_START),
    'loss_beta_end':    float(LOSS_BETA_END),
    'lambda_dist':      float(LAMBDA_DIST),
    'lambda_dfl':       float(LAMBDA_DFL),
    'lambda_anchor':    float(LAMBDA_ANCHOR_EFF),
    'fusion_mode':      FUSION_MODE,
    'is_late_fusion':   int(IS_LATE_FUSION),
    'mod_drop_p':       (float(np.mean(list(MOD_DROP_P.values())))
                        if isinstance(MOD_DROP_P, dict) else float(MOD_DROP_P)),
    'aux_loss_weight':  float(AUX_LOSS_WEIGHT),
    'aux_loss_n_modalities': int(len(AUX_LOSS_MODALITIES)),
    'tta_phase_k':      int(tta_k),
    'ema_decay':        float(EMA_DECAY),
    'aug_noise_p':      float(AUG_NOISE_P),
    'experiment_name':  EXPERIMENT_NAME,
    'scenario_name':    SCENARIO_NAME,
    'run_mode':         RUN_MODE,
    'arch_tag':         ARCH_TAG,
    'modalities':       ", ".join(mod_list),
}
if RUN_MODE == 'train_and_eval':
    mat_dict.update({
        'best_val_mean_error_m': float(best_val_err),
        'best_epoch':            int(best_epoch),
        'epochs_trained':        int(epoch),
        'early_stopped':         int(epochs_no_improve >= PATIENCE),
        'train_time_min':        float(total_time / 60.0),
    })
    if USE_ATTN_REFINE and not IS_LATE_FUSION:
        mat_dict['refine_scale_absmean'] = float(model.refine_scale.detach().abs().mean())
    if IS_LATE_FUSION:
        mat_dict['pool_gain_absmean'] = float(model.pool.net[-1].weight.detach().abs().mean())
        for _k in ('csi', 'cam', 'conflict', 'prec_csi'):
            if history.get('gate_mean', {}).get(_k):
                mat_dict[f'gate_{_k}_absmean'] = float(np.mean(history['gate_mean'][_k]))

mat_path = OUTPUT_DIR / "results.mat"
sio.savemat(mat_path, mat_dict, do_compression=True)
print(f"[OK] MATLAB-friendly  -> {mat_path}")

results_json_dict = {
    'experiment':  EXPERIMENT_NAME,
    'run_mode':    RUN_MODE,
    'arch_tag':    ARCH_TAG,
    'scenario':    SCENARIO_NAME,
    'modalities':  mod_list,
    'metrics': {
        'mean_error_3d_m':   mean_error_3d,
        'median_error_3d_m': median_error_3d,
        'rmse_3d_m':         rmse_3d,
        'p90_error_3d_m':    p90_error_3d,
        'p95_error_3d_m':    p95_error_3d,
        'mean_error_2d_m':   mean_error_2d,
        'median_error_2d_m': median_error_2d,
        'rmse_2d_m':         rmse_2d,
        'p90_error_2d_m':    p90_error_2d,
        'p95_error_2d_m':    p95_error_2d,
        'mae_per_axis_m':    err_per_axis_mae.tolist(),
        'acc_within_1m_pct': acc_1m,
        'acc_within_2m_pct': acc_2m,
        'acc_within_5m_pct': acc_5m,
    },
    'params':       int(train_params),
    'total_params': int(total_params),
    'flops':        model_flops,
    'config': {
        'use_csi': USE_CSI, 'use_bs_pos': USE_BS_POS,
        'use_camera': USE_CAMERA, 'use_lidar': USE_LIDAR,
        'use_all_cameras': USE_ALL_CAMERAS,
        'batch_size': BATCH_SIZE, 'epochs': EPOCHS, 'lr': LR,
        'weight_decay': WEIGHT_DECAY, 'patience': PATIENCE,
        'warmup_epochs': WARMUP_EPOCHS, 'min_lr': MIN_LR, 'seed': SEED,
        'csi_embed_dim': CSI_EMBED_DIM, 'pos_embed_dim': POS_EMBED_DIM,
        'cam_embed_dim': CAM_EMBED_DIM, 'lidar_embed_dim': LIDAR_EMBED_DIM,
        'fusion_dim': FUSION_DIM, 'csi_width': CSI_WIDTH, 'csi_blocks': CSI_BLOCKS,
        'use_attn_refine': USE_ATTN_REFINE, 'attn_dim': ATTN_DIM,
        'num_heads': NUM_HEADS, 'num_fusion_tokens': NUM_FUSION_TOKENS,
        'num_fusion_layers': NUM_FUSION_LAYERS, 'num_lidar_tokens': NUM_LIDAR_TOKENS,
        'attn_lr_mult': ATTN_LR_MULT,
        'use_dist_head': USE_DIST_HEAD, 'num_bins': NUM_BINS, 'bin_range': BIN_RANGE,
        'dropout': DROPOUT, 'grad_clip': GRAD_CLIP,
        'num_ant': NUM_ANT, 'num_sub': NUM_SUB,
        'img_size': IMG_SIZE, 'lidar_points': LIDAR_POINTS,
        'out_dim': OUT_DIM,
        'csi_std_scenario': float(CSI_STD), 'csi_feat_ch': CSI_FEAT_CH,
        'loss_beta_start': LOSS_BETA_START, 'loss_beta_end': LOSS_BETA_END,
        'lambda_dist': LAMBDA_DIST, 'lambda_dfl': LAMBDA_DFL,
        'lambda_anchor': LAMBDA_ANCHOR_EFF, 'fusion_mode': FUSION_MODE,
        'is_late_fusion': bool(IS_LATE_FUSION),
        'mod_drop_p': MOD_DROP_P, 'tta_phase_k': tta_k,
        'aux_loss_modalities': AUX_LOSS_MODALITIES, 'aux_loss_weight': AUX_LOSS_WEIGHT,
        'ema_decay': EMA_DECAY,
        'aug_phase': AUG_PHASE, 'aug_noise_p': AUG_NOISE_P,
        'aug_snr_db': list(AUG_SNR_DB),
        'amp_dtype': str(AMP_DTYPE),
    },
    'data': {
        'hdf5':                str(HDF5_FILE),
        'valid_samples_json':  str(VALID_SAMPLES_FILE),
        'clean_indices_file':  str(CLEAN_INDICES_FILE),
        'gps_stats':           str(GPS_STATS_FILE),
        'gps_mean':            GPS_MEAN.tolist(),
        'gps_std':             GPS_STD.tolist(),
        'gps_min':             GPS_MIN.tolist(),
        'gps_max':             GPS_MAX.tolist(),
        'offset_stats_file':   str(OFFSET_STATS_FILE),
        'csi_logstats_file':   str(CSI_LOGSTATS_FILE),
        'off_mean':            OFF_MEAN.tolist(),
        'off_std':             OFF_STD.tolist(),
        'off_scale':           float(OFF_SCALE),
        'axis_scale':          AXIS_SCALE.tolist(),
        'hdf5_total':          int(num_samples),
        'clean_total':         int(len(CLEAN_INDICES)),
        'train':               len(train_idx),
        'val':                 len(val_idx),
        'test':                len(test_idx),
    },
}
if RUN_MODE == 'train_and_eval':
    results_json_dict.update({
        'best_val_mean_error_m': float(best_val_err),
        'best_epoch':            int(best_epoch),
        'epochs_trained':        int(epoch),
        'early_stopped':         bool(epochs_no_improve >= PATIENCE),
        'time_min':              round(total_time / 60.0, 2),
    })
    if USE_ATTN_REFINE and not IS_LATE_FUSION:
        results_json_dict['refine_scale_absmean'] = float(
            model.refine_scale.detach().abs().mean())
    if IS_LATE_FUSION:
        results_json_dict['pool_gain_absmean'] = float(
            model.pool.net[-1].weight.detach().abs().mean())
with open(OUTPUT_DIR / "results.json", 'w') as f:
    json.dump(results_json_dict, f, indent=2)
print(f"[OK] JSON results     -> {OUTPUT_DIR}/results.json")


txt_lines = []
def _w(s=""): txt_lines.append(s)

_w("=" * 78)
_w(" MULTIMODAL GPS LOCALIZATION  --  EXPERIMENT REPORT ")
_w("=" * 78)
_w(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
_w(f"Experiment name : {EXPERIMENT_NAME}")
_w(f"Scenario        : {SCENARIO_NAME}  ({SCENARIO_FOLDER})")
_w(f"Run mode        : {RUN_MODE}")
_w("")
_w("--- Modalities -----------------------------------------------------------")
_w(f"  CSI            : {USE_CSI}")
_w(f"  BS position    : {USE_BS_POS}")
_w(f"  Camera         : {USE_CAMERA}"
   + (f"  (cams={NUM_CAMERAS if USE_ALL_CAMERAS else 1}/BS, total={NUM_CAMERAS})" if USE_CAMERA else ""))
_w(f"  LiDAR          : {USE_LIDAR}"
   + (f"  ({LIDAR_POINTS} pts -> {NUM_LIDAR_TOKENS} PMA tokens)" if USE_LIDAR else ""))
_w("")
_w("--- Architecture (v5 DFR-Fusion) -----------------------------------------")
_w(f"  CSI features      : {CSI_FEAT_CH} ch (Re, Im, log|h|, cos/sin phase,")
_w(f"                      log|A|, cos/sin angle(A), cos/sin d_sub, cos/sin d_ant)")
_w(f"  CSI trunk         : {CSI_WIDTH} ch x {CSI_BLOCKS} residual blocks, stride 1, no pooling")
_w(f"  CSI dense readout : Flatten({CSI_WIDTH * NUM_ANT * NUM_SUB}) -> 768 -> {CSI_EMBED_DIM}")
_w(f"  Dense fusion trunk: {FUSION_DIM}")
_w(f"  Attn refinement   : {USE_ATTN_REFINE}  ({NUM_FUSION_TOKENS} latent tokens x "
   f"{NUM_FUSION_LAYERS} layers, dim {ATTN_DIM}, {NUM_HEADS} heads)")
_w(f"                      added through a LayerScale INITIALIZED TO ZERO")
_w(f"  Distributional hd : {USE_DIST_HEAD}  ({NUM_BINS} bins/axis, +-{BIN_RANGE} axis-std)")
_w(f"  Physics anchor    : per-sample-standardized beamspace soft-argmax,")
_w(f"                      {NUM_ANCHOR_STATS} stats, zero-init prior, "
   f"lambda={LAMBDA_ANCHOR_EFF}"
   + ("  (DISABLED: encoders frozen)" if IS_LATE_FUSION else ""))
_w(f"  Fusion mode       : {FUSION_MODE}"
   + ("  (v16 two-stage: frozen experts + CADPool)" if IS_LATE_FUSION
      else "  (joint end-to-end feature fusion)"))
if IS_LATE_FUSION:
    _w(f"  CADPool           : conflict-aware distributional log-opinion pool")
    _w(f"                      p_fused ~ p_csi^a * p_cam^b, a/b per-sample AND")
    _w(f"                      per-axis, from {CADPool.N_DESC} descriptors/axis")
    _w(f"                      (mu, sd, entropy, residual of each expert +")
    _w(f"                      Jensen-Shannon divergence + standardized mean gap)")
    _w(f"                      zero-init => epoch 0 IS the precision-weighted")
    _w(f"                      (inverse-variance) classical pool")
    _w(f"                      v24: gate ALSO sees each expert's penultimate")
    _w(f"                      embedding (dim {CSI_EMBED_DIM}/{CAM_EMBED_DIM}), compressed to "
       f"{LATE_EMB_HIDDEN} dims each")
    _w(f"                      before concatenation -- previously computed inside each")
    _w(f"                      expert and discarded before reaching the combiner")
    _w(f"                      trainable {sum(p.numel() for p in model.pool.parameters()):,} / "
       f"frozen {frozen_params:,}")
    _w(f"                      gate clamp +-{LATE_GATE_CLAMP}, corr scale {LATE_CORR_SCALE}, "
       f"mod-drop {LATE_MOD_DROP_P}")
    try:
        _w(f"  Frozen CSI expert : {find_base_checkpoint('csi').parent.name}")
        _w(f"  Frozen cam expert : {find_base_checkpoint('cam').parent.name}")
    except Exception:
        pass
_w(f"  dropout           : {DROPOUT}")
_w(f"  modality dropout  : "
   f"{', '.join(f'{k}={v}' for k, v in MOD_DROP_P.items()) if isinstance(MOD_DROP_P, dict) else f'p={MOD_DROP_P}'}")
_w(f"  LayerScale init   : {LS_INIT}  (attention LR x{ATTN_LR_MULT})")
_w(f"  Aux modality loss : modalities={AUX_LOSS_MODALITIES}  "
   f"weight={AUX_LOSS_WEIGHT}")
_w(f"  CSI confidence gt : {USE_CSI_CONFIDENCE_GATE}  "
   f"(GMU-style multiplicative pre-trunk gate, Arevalo et al. ICLR 2017 WS)")
_w(f"  Cam confidence gt : {USE_CAM_CONFIDENCE_GATE}  (symmetric mirror of "
   f"the CSI gate)")
_w(f"  TTA               : {tta_k} phase rotations at test time")
_w(f"  output dim        : {OUT_DIM}  (BS-relative offset, arch={ARCH_TAG})")
_w(f"  CSI std (scen.)   : {CSI_STD:.6e}")
_w(f"  Trainable params  : {train_params:,}")
_w(f"  Total params      : {total_params:,}")
if model_flops is not None:
    _w(f"  FLOPs / sample    : {model_flops:,}  (~{model_flops/1e6:.2f} M)")
else:
    _w(f"  FLOPs / sample    : (fvcore not available)")
if RUN_MODE == 'train_and_eval' and USE_ATTN_REFINE and not IS_LATE_FUSION:
    _w(f"  LayerScale |gamma|: {float(model.refine_scale.detach().abs().mean()):.5f} "
       f"(0.0 == stayed at the dense trunk)")
if RUN_MODE == 'train_and_eval' and IS_LATE_FUSION:
    _w(f"  CADPool |weight|  : "
       f"{float(model.pool.net[-1].weight.detach().abs().mean()):.5f} "
       f"(0.0 == stayed at the zero-init classical estimator)")
_w("")
_w("--- Training set-up ------------------------------------------------------")
_w(f"  batch_size     : {BATCH_SIZE}")
_w(f"  epochs (max)   : {EPOCHS}")
_w(f"  optimizer      : AdamW, 4 param groups (decay on >=2-D weights only)")
_w(f"  lr             : {LR}   (attention params x{ATTN_LR_MULT})")
_w(f"  weight_decay   : {WEIGHT_DECAY}")
_w(f"  scheduler      : cosine annealing with linear warmup")
_w(f"  warmup_epochs  : {WARMUP_EPOCHS}")
_w(f"  min_lr         : {MIN_LR}")
_w(f"  patience       : {PATIENCE}")
_w(f"  grad clip      : {GRAD_CLIP}")
_w(f"  loss           : SmoothL1(beta {LOSS_BETA_START} -> {LOSS_BETA_END}, annealed)")
_w(f"                   + {LAMBDA_DIST}*mean||.||2 + {LAMBDA_DFL}*DFL "
   f"+ {LAMBDA_ANCHOR}*anchor")
_w(f"  target scaling : ISOTROPIC, OFF_SCALE = {OFF_SCALE:.4f} m")
_w(f"                   (1 loss unit == {OFF_SCALE:.4f} m on every axis, so the")
_w(f"                    Euclidean term IS the reported mean distance error)")
_w(f"  OFF_STD        : {OFF_STD.tolist()}   (per-axis; reporting / bin grid only)")
_w(f"  EMA decay      : {EMA_DECAY}  (bias-corrected; validated/checkpointed weights)")
_w(f"  augmentation   : phase-rot={AUG_PHASE}, AWGN p={AUG_NOISE_P}, "
   f"SNR {AUG_SNR_DB[0]:.0f}-{AUG_SNR_DB[1]:.0f} dB (train only)")
_w(f"  AMP            : {AMP_DTYPE} (physics anchor computed in float32)")
_w(f"  seed           : {SEED}")
_w("")
if RUN_MODE == 'train_and_eval':
    _w(f"  Best epoch     : {best_epoch}")
    _w(f"  Best val mean error : {best_val_err:.3f} m")
    _w(f"  Epochs trained : {epoch}")
    _w(f"  Early stopped  : {epochs_no_improve >= PATIENCE}")
    _w(f"  Total time     : {total_time/60.0:.2f} min")
    _w("")
_w("--- Dataset --------------------------------------------------------------")
_w(f"  Samples in HDF5    : {num_samples:,}")
_w(f"  Clean samples used : {len(CLEAN_INDICES):,}  "
   f"(dropped {num_samples - len(CLEAN_INDICES):,} with missing UE/BS positions)")
_w(f"  Train / Val / Test : {len(train_idx):,} / {len(val_idx):,} / {len(test_idx):,}")
_w(f"  HDF5 cache         : {HDF5_FILE}")
_w(f"  Valid samples      : {VALID_SAMPLES_FILE}")
_w(f"  Clean indices      : {CLEAN_INDICES_FILE}")
_w(f"  GPS mean           : {GPS_MEAN.tolist()}")
_w(f"  GPS std            : {GPS_STD.tolist()}")
_w(f"  GPS min            : {GPS_MIN.tolist()}")
_w(f"  GPS max            : {GPS_MAX.tolist()}")
_w(f"  OFFSET mean        : {OFF_MEAN.tolist()}")
_w(f"  OFFSET scale       : {OFF_SCALE:.6f}  (scalar)")
_w(f"  AXIS scale         : {AXIS_SCALE.tolist()}")
_w("")
_w("--- Test-set positioning error (metres) ----------------------------------")
_w("  3D Euclidean error:")
_w(f"    mean            : {mean_error_3d:.4f}")
_w(f"    median          : {median_error_3d:.4f}")
_w(f"    RMSE            : {rmse_3d:.4f}")
_w(f"    90th percentile : {p90_error_3d:.4f}")
_w(f"    95th percentile : {p95_error_3d:.4f}")
_w("  2D (x-y) error:")
_w(f"    mean            : {mean_error_2d:.4f}")
_w(f"    median          : {median_error_2d:.4f}")
_w(f"    RMSE            : {rmse_2d:.4f}")
_w(f"    90th percentile : {p90_error_2d:.4f}")
_w(f"    95th percentile : {p95_error_2d:.4f}")
_w("  Per-axis MAE       :")
_w(f"    x : {err_per_axis_mae[0]:.4f}")
_w(f"    y : {err_per_axis_mae[1]:.4f}")
_w(f"    z : {err_per_axis_mae[2]:.4f}")
_w("  Accuracy within threshold (3D):")
_w(f"    <= 1 m : {acc_1m:.2f} %")
_w(f"    <= 2 m : {acc_2m:.2f} %")
_w(f"    <= 5 m : {acc_5m:.2f} %")
_w("")
_w("--- Training history (last entries) --------------------------------------")
if len(history.get('train_loss', [])) > 0:
    nshow = min(10, len(history['train_loss']))
    _has_refine = len(history.get('refine_scale_absmean', [])) > 0
    _has_pool_g = len(history.get('pool_gain_absmean', [])) > 0
    _has_aux = any(v == v for v in history.get('aux_loss', []))   # any non-NaN
    _gate_mods = [m for m in history.get('gate_mean', {}) if history['gate_mean'][m]]
    _hdr = "epoch, train_loss, val_loss, val_mean_err_m, lr, beta"
    if _has_aux:
        _hdr += ", aux_loss"
    if _has_refine:
        _hdr += ", |gamma|"
    if _has_pool_g:
        _hdr += ", |pool_g|"
    _hdr += "".join(f", gate[{m}]" for m in _gate_mods)
    _w(f"  Showing last {nshow} epochs ({_hdr}):")
    start = len(history['train_loss']) - nshow
    _betas = history.get('beta', [])
    for i in range(start, len(history['train_loss'])):
        _b = _betas[i] if i < len(_betas) else float('nan')
        line = (f"   {i+1:4d} {history['train_loss'][i]:.6f} {history['val_loss'][i]:.6f} "
               f"{history['val_mean_error_m'][i]:.4f} {history['lr'][i]:.3e} {_b:.4f}")
        if _has_aux:
            al_i = history['aux_loss'][i] if i < len(history['aux_loss']) else float('nan')
            line += f" {al_i:.6f}"
        if _has_refine:
            rs_i = history['refine_scale_absmean'][i] if i < len(history['refine_scale_absmean']) else float('nan')
            line += f" {rs_i:.4f}"
        if _has_pool_g:
            pg_i = history['pool_gain_absmean'][i] if i < len(history['pool_gain_absmean']) else float('nan')
            line += f" {pg_i:.4f}"
        for m in _gate_mods:
            gv = history['gate_mean'][m][i] if i < len(history['gate_mean'][m]) else float('nan')
            line += f" {gv:.3f}"
        _w(line)
_w("")
_w("--- Files written --------------------------------------------------------")
_w(f"  {best_model_path.name}")
if RUN_MODE == 'train_and_eval':
    _w(f"  {final_model_path.name}")
    _w(f"  training_history.json / .csv")
_w(f"  results.mat   (variables: predictions, errors, history, hyper-params)")
_w(f"  results.json  (machine-readable metrics)")
_w(f"  results.txt   (this file)")
_w(f"  split_indices.json")
_w(f"  curves.png / curves.pdf")
_w(f"  cdf_error.png / cdf_error.pdf")
_w("=" * 78)

with open(OUTPUT_DIR / "results.txt", 'w') as f:
    f.write("\n".join(txt_lines))
print(f"[OK] Human-readable   -> {OUTPUT_DIR}/results.txt")


if len(history.get('train_loss', [])) > 0:
    _has_refine = len(history.get('refine_scale_absmean', [])) > 0
    _has_pool_g = len(history.get('pool_gain_absmean', [])) > 0
    _has_gates = any(len(v) > 0 for v in history.get('gate_mean', {}).values())
    n_panels = 4 if (_has_refine or _has_pool_g or _has_gates) else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(16 if n_panels == 3 else 21, 4.5))
    ep = range(1, len(history['train_loss']) + 1)

    axes[0].plot(ep, history['train_loss'], 'b-', alpha=0.8, label='Train')
    axes[0].plot(ep, history['val_loss'],   'r-', alpha=0.8, label='Val')
    if any(v == v for v in history.get('aux_loss', [])):   # any non-NaN
        axes[0].plot(ep, history['aux_loss'], color='teal', alpha=0.7,
                    ls='--', label='Aux (modality-competition fix)')
    axes[0].set(xlabel='Epoch', ylabel='DFR loss (scaled offset)', title='Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(ep, history['val_mean_error_m'], 'g-', alpha=0.8, label='Val mean err')
    axes[1].axhline(mean_error_3d, color='red', ls='--',
                    label=f'Test mean {mean_error_3d:.2f} m')
    axes[1].axhline(1.0, color='k', ls='-.', alpha=0.4, label='1 m target')
    if RUN_MODE == 'train_and_eval':
        axes[1].axvline(best_epoch, color='gray', ls=':', alpha=0.5,
                        label=f'Best ep {best_epoch}')
    axes[1].set(xlabel='Epoch', ylabel='Mean 3D error (m)', title='Positioning error')
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    axes[2].plot(ep, history['lr'], color='orange', alpha=0.8, label='LR')
    axes[2].set(xlabel='Epoch', ylabel='LR', title='LR (cosine + warmup) & SmoothL1 beta')
    axes[2].set_yscale('log'); axes[2].grid(True, alpha=0.3)
    if len(history.get('beta', [])) == len(history['train_loss']):
        ax2 = axes[2].twinx()
        ax2.plot(ep, history['beta'], color='purple', alpha=0.7, ls='--', label='beta')
        ax2.set_ylabel('SmoothL1 beta', color='purple')
        ax2.tick_params(axis='y', labelcolor='purple')

    if _has_refine or _has_pool_g or _has_gates:
        if _has_refine:
            axes[3].plot(ep, history['refine_scale_absmean'], color='teal',
                        alpha=0.9, label='|gamma| (refine_scale)')
        if _has_pool_g:
            axes[3].plot(ep, history['pool_gain_absmean'], color='teal',
                        alpha=0.9, label='|weight| (CADPool last layer)')
        _panel_subject = ('CADPool' if _has_pool_g else
                          'Attention branch' if _has_refine else 'Gating')
        axes[3].set(xlabel='Epoch', ylabel='mean |departure from zero-init|',
                   title=f'{_panel_subject} & confidence gates')
        axes[3].grid(True, alpha=0.3)
        gate_colors = {'csi': 'crimson', 'cam': 'steelblue', 'lidar': 'darkorange',
                       'csi_confidence': 'darkviolet', 'cam_confidence': 'goldenrod',
                       'conflict': 'black', 'prec_csi': 'seagreen'}
        gm_dict = history.get('gate_mean', {})
        if _has_gates:
            ax3 = axes[3].twinx()
            for m, vals in gm_dict.items():
                if vals:
                    ax3.plot(ep, vals, color=gate_colors.get(m, 'gray'),
                            alpha=0.7, ls='--', label=f'gate[{m}]')
            ax3.axhline(0.5, color='gray', ls=':', alpha=0.4)
            ax3.set_ylabel('per-modality reliability gate (0-1)')
            ax3.set_ylim(0, 1)
            h1, l1 = axes[3].get_legend_handles_labels()
            h2, l2 = ax3.get_legend_handles_labels()
            axes[3].legend(h1 + h2, l1 + l2, fontsize=7, loc='upper left')
        else:
            axes[3].legend(fontsize=8)

    plt.suptitle(EXPERIMENT_NAME, fontsize=8, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "curves.png", dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "curves.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Training curves -> {OUTPUT_DIR}/curves.{{png,pdf}}")

fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
sorted_3d = np.sort(err_3d)
sorted_2d = np.sort(err_2d)
cdf = np.linspace(0, 1, len(sorted_3d), endpoint=False)
ax.plot(sorted_3d, cdf, 'b-', label=f'3D (mean={mean_error_3d:.2f} m)')
ax.plot(sorted_2d, cdf, 'g-', label=f'2D (mean={mean_error_2d:.2f} m)')
ax.axvline(p90_error_3d, color='b', ls=':', alpha=0.5)
ax.axvline(p90_error_2d, color='g', ls=':', alpha=0.5)
ax.axvline(1.0, color='k', ls='-.', alpha=0.4, label=f'1 m ({acc_1m:.1f} % of samples)')
ax.set(xlabel='Positioning error (m)', ylabel='CDF',
       title=f'Positioning error CDF -- {SCENARIO_NAME}')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cdf_error.png", dpi=150, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / "cdf_error.pdf", bbox_inches='tight')
plt.close(fig)
print(f"[OK] Error CDF       -> {OUTPUT_DIR}/cdf_error.{{png,pdf}}")
