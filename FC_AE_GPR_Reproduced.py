import os
import io
import sys
import json
import math
import time
import shutil
import warnings
from pathlib import Path

import numpy as np
import h5py

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        desc = kw.get('desc', '')
        total = kw.get('total', None)
        try:
            total = total if total is not None else len(it)
        except TypeError:
            total = None
        for i, x in enumerate(it):
            if total and i % max(1, total // 20) == 0:
                print(f"    {desc} {i}/{total}", flush=True)
            yield x

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

warnings.filterwarnings('ignore')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[OK] Imports done. DEVICE = {DEVICE}")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/DATA/tanmoy/Documents/deepverse_localization")
CACHE_DIR    = PROJECT_ROOT / "Cache"
RESULTS_DIR  = PROJECT_ROOT / "Results_FCAEGPR"

SCENARIO_NAME = 'DT31'          # any DeepVerse scenario

SCENARIO_REGISTRY = {
    'Carla-Town01': {'folder': 'Carla-Town01', 'total_scenes': 2200,
                     'num_bs': 5, 'num_ue_max': 200, 'cams_per_bs': 4, 'lidars_per_bs': 1},
    'Carla-Town05': {'folder': 'Carla-Town05', 'total_scenes': 4000,
                     'num_bs': 3, 'num_ue_max': 200, 'cams_per_bs': 5, 'lidars_per_bs': 1},
    'O1':           {'folder': 'O1', 'total_scenes': 2000,
                     'num_bs': 4, 'num_ue_max': 100, 'cams_per_bs': 3, 'lidars_per_bs': 1},
    'I1':           {'folder': 'I1', 'total_scenes': 2000,
                     'num_bs': 2, 'num_ue_max': 20, 'cams_per_bs': 0, 'lidars_per_bs': 1},
    'DT1':          {'folder': 'DT1', 'total_scenes': 2411,
                     'num_bs': 1, 'num_ue_max': 5, 'cams_per_bs': 1, 'lidars_per_bs': 1},
    'DT31':         {'folder': 'DT31', 'total_scenes': 7012,
                     'num_bs': 1, 'num_ue_max': 5, 'cams_per_bs': 1, 'lidars_per_bs': 1},
}
assert SCENARIO_NAME in SCENARIO_REGISTRY, f"Unknown scenario '{SCENARIO_NAME}'"
_cfg               = SCENARIO_REGISTRY[SCENARIO_NAME]
SCENARIO_FOLDER    = _cfg['folder']
TOTAL_SCENES       = _cfg['total_scenes']
NUM_BS             = _cfg['num_bs']
NUM_UE_MAX         = _cfg['num_ue_max']
NUM_CAMERAS_PER_BS = _cfg['cams_per_bs']
NUM_LIDARS_PER_BS  = _cfg['lidars_per_bs']

SCENARIO_PATH = str(PROJECT_ROOT / "scenarios" / SCENARIO_FOLDER)
CONFIG_PATH   = f"{SCENARIO_PATH}/param/config1.m"

# DT31 ships with only [0:7] of 512 subcarriers selected -- too coarse for
# ranging. spreading ADP_NC bins across the full band instead matches the
# paper's OFDM grid spacing.
OFDM_TOTAL_SUBCARRIERS = 512     # dv.comm.OFDM.subcarriers
OFDM_BANDWIDTH_HZ      = 0.05e9  # dv.comm.OFDM.bandwidth, for the delay report only
SUBCARRIER_MODE        = 'spread'   # 'spread' (recommended) | 'contiguous'
SUBCARRIER_STRIDE      = None       # None -> OFDM_TOTAL/Nc

# --- cache rebuild flags ---
FORCE_REBUILD_CSI     = False
FORCE_RESCAN_SAMPLES  = False
FORCE_REBUILD_ADP     = False
REUSE_V6_VALID_SAMPLES = True    # read-only reuse of v6's valid_samples.json

# --- ADP geometry / antenna array ---
# ADP = |V^H H F|, shape (n_bs_antennas, n_subcarriers). presets mirror paper
# Table II/VIII (Nt=Nc = 4/8/16/32/64). ULA only
# angle axis.
ADP_PRESET = 'paper16'

_ADP_PRESETS = {
    'paper4':  dict(adp_nt=4,  adp_nc=4,  ant_shape=(4, 1)),
    'paper8':  dict(adp_nt=8,  adp_nc=8,  ant_shape=(8, 1)),
    'paper16': dict(adp_nt=16, adp_nc=16, ant_shape=(16, 1)),
    'paper32': dict(adp_nt=32, adp_nc=32, ant_shape=(32, 1)),
    'paper64': dict(adp_nt=64, adp_nc=64, ant_shape=(64, 1)),
    'native':  dict(adp_nt=16, adp_nc=8,  ant_shape=(16, 1)),
}
assert ADP_PRESET in _ADP_PRESETS, f"ADP_PRESET must be one of {list(_ADP_PRESETS)}"
_p = _ADP_PRESETS[ADP_PRESET]
ADP_NT, ADP_NC = _p['adp_nt'], _p['adp_nc']
ADP_TAG   = f"adp{ADP_NT}x{ADP_NC}"

ANT_SHAPE = _p['ant_shape']
ANT_TAG   = f"ant{ANT_SHAPE[0]}_{ANT_SHAPE[1]}"
assert ANT_SHAPE[1] == 1 or ANT_SHAPE[0] == 1, (
    "The paper's method is defined for a ULA (num_ant_x=1, num_ant_z=1). "
    "A UPA would make the ADP's angle axis physically meaningless -- the "
    "Nt-point DFT V in Eq. (2) is a 1-D array-response transform.")

NUM_ANT = ANT_SHAPE[0] * ANT_SHAPE[1]
NUM_SUB = ADP_NC
assert ADP_NT <= NUM_ANT, f"ADP_NT={ADP_NT} exceeds the {NUM_ANT}-element array"
assert NUM_SUB <= OFDM_TOTAL_SUBCARRIERS, (
    f"Need {NUM_SUB} subcarriers but the scenario's OFDM grid has only "
    f"{OFDM_TOTAL_SUBCARRIERS}. Check dv.comm.OFDM.subcarriers in the config.")

RAW_GPR_MAX_DIM = 256
RUN_FCAE        = (ADP_NT * ADP_NC) > 16
RUN_RAW_GPR     = (ADP_NT * ADP_NC) <= RAW_GPR_MAX_DIM

if SUBCARRIER_MODE == 'spread':
    _stride = (SUBCARRIER_STRIDE if SUBCARRIER_STRIDE is not None
               else OFDM_TOTAL_SUBCARRIERS // NUM_SUB)
    assert _stride >= 1, "SUBCARRIER_STRIDE must be >= 1"
    SELECTED_SUBCARRIERS = [i * _stride for i in range(NUM_SUB)]
elif SUBCARRIER_MODE == 'contiguous':
    _stride = 1
    SELECTED_SUBCARRIERS = list(range(NUM_SUB))
else:
    raise ValueError("SUBCARRIER_MODE must be 'spread' or 'contiguous'")
assert len(SELECTED_SUBCARRIERS) == NUM_SUB
assert max(SELECTED_SUBCARRIERS) < OFDM_TOTAL_SUBCARRIERS, (
    f"Subcarrier index {max(SELECTED_SUBCARRIERS)} exceeds the "
    f"{OFDM_TOTAL_SUBCARRIERS}-point OFDM grid. Lower SUBCARRIER_STRIDE.")

_C = 299_792_458.0
_spacing_hz = _stride * (OFDM_BANDWIDTH_HZ / OFDM_TOTAL_SUBCARRIERS)
_span_hz    = (NUM_SUB - 1) * _spacing_hz
DELAY_RES_M   = (_C / _span_hz) if _span_hz > 0 else float('inf')
UNAMBIG_M     = (_C / _spacing_hz) if _spacing_hz > 0 else float('inf')

# --- protocol, matches authors' GPR_train_kfold.m / GPR_test_kfold.m ---
PERCENT_ALL   = [10, 5, 1, 0.5, 0.25, 0.1]
N_FOLDS       = 5
TEST_FRACTION = 0.10
TEST_SIZE     = 1001

# --- FC-AE, matches authors' FC_AE_Models.ipynb ---
AUG_PERCENT   = 0.005
# authors' script takes 0.5% of the set -> 362 unique ADPs on O1, but only 35
# on DT31 (7,012 samples) -- not enough diversity. AUG_N_SOURCE pins the
# absolute count instead so it matches what the paper actually used.
AUG_N_SOURCE  = 362
AUG_MAX_SAMPLES = 60000  
FCAE_EPOCHS   = 500      
FCAE_BATCH    = 32
FCAE_LR       = 5e-4
FCAE_VAL_SPLIT= 0.2
LEAKY_SLOPE   = 0.3      

# --- GPR, matches authors' GPR_train_kfold.m ---
GPR_LENGTH_SCALE = 0.01  
GPR_SIGMA_F      = 25.0  #
GPR_SIGMA_N      = 0.5   
GPR_MEDIAN_HEURISTIC = True   
GPR_RESTARTS         = 2
GPR_LS_BOUND_FACTOR  = 100.0  

# --- training set sizes ---

TRAIN_SIZE_MODE = 'percent'
TRAIN_SIZES_ABS = [3620, 724, 362, 181, 72]

# --- CNN benchmark, matches authors' CNN_Localization.ipynb ---
CNN_EPOCHS    = 1000
CNN_BATCH     = 100
CNN_VAL_SPLIT = 0.1
CNN_PATIENCE  = 50

RANDOM_SEED   = 42

# --- ADP scaling ---

ADP_SCALE_MODE = 'std'     

CNN_NORMALIZE_TARGETS = True

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_SC = SCENARIO_FOLDER.lower().replace('-', '_')
SUB_TAG = f"sub{NUM_SUB}{'s' if SUBCARRIER_MODE == 'spread' else 'c'}"


OWN_PREFIX         = f"fcaegpr_{_SC}_{ANT_TAG}_{SUB_TAG}"
CSI_H5_FILE        = CACHE_DIR / f"{OWN_PREFIX}_csi.h5"
OWN_SAMPLES_FILE   = CACHE_DIR / f"fcaegpr_{_SC}_valid_samples.json"
ADP_CACHE_FILE     = CACHE_DIR / f"{OWN_PREFIX}_{ADP_TAG}_ADP.npz"

V6_SAMPLES_FILE    = CACHE_DIR / f"{_SC}_valid_samples.json"   # read-only

CACHE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 72)
print(f"  SCENARIO   : {SCENARIO_NAME}  ({SCENARIO_FOLDER})  {TOTAL_SCENES} scenes")
print(f"  ANT_SHAPE  : {ANT_SHAPE}  [{ANT_TAG}]  (ULA, per the paper)")
print(f"  SUBCARRIER : {NUM_SUB} of {OFDM_TOTAL_SUBCARRIERS}, mode='{SUBCARRIER_MODE}'")
print(f"               selected = {SELECTED_SUBCARRIERS[:4]}"
      + (f" ... {SELECTED_SUBCARRIERS[-1]}" if NUM_SUB > 4 else ""))
print(f"  ADP_PRESET : '{ADP_PRESET}'  ->  ADP {ADP_NT} x {ADP_NC}")
print(f"  DELAY      : resolution ~{DELAY_RES_M:.1f} m, "
      f"unambiguous range ~{UNAMBIG_M:.1f} m")
if UNAMBIG_M < 150.0:
    print(f"  !! WARNING: unambiguous range {UNAMBIG_M:.0f} m may be shorter "
          f"than the BS-UE distances in this scenario.")
    print(f"     Delays beyond it alias, weakening the smooth-similarity prior "
          f"the GPR and correlation baselines rely on.")
    print(f"     If your ranges exceed ~{UNAMBIG_M:.0f} m, set "
          f"SUBCARRIER_STRIDE to something smaller than {_stride} "
          f"(trades delay resolution for range).")
print(f"  METHODS    : FC-AE={'on' if RUN_FCAE else 'OFF (ADP already 4x4)'}, "
      f"raw-GPR={'on' if RUN_RAW_GPR else f'OFF (dim {ADP_NT*ADP_NC} > {RAW_GPR_MAX_DIM})'}")
print(f"  CSI cache  : {CSI_H5_FILE.name}   (exclusive to this script)")
print(f"  ADP cache  : {ADP_CACHE_FILE.name}")
print(f"  FOLDS      : {N_FOLDS} (authors: 50)")
print(f"  REBUILD    : csi={FORCE_REBUILD_CSI}  adp={FORCE_REBUILD_ADP}  "
      f"samples={FORCE_RESCAN_SAMPLES}")
print("=" * 72)


# ---------------------------------------------------------------------------
# Experiment naming
# ---------------------------------------------------------------------------
def get_experiment_name() -> str:
    return f"fcaegpr_{_SC}_{ANT_TAG}_{ADP_TAG}_{N_FOLDS}fold"

EXP_NAME = get_experiment_name()
print(f"[OK] Experiment: {EXP_NAME}")



# ---------------------------------------------------------------------------
def _set_nested(params, path_options, value, what):
    """Try each candidate key path in turn, set the first one that exists.
    Belt and braces -- _verify_dv_shapes() below is the real check."""
    applied = []
    for path in path_options:
        node = params
        ok = True
        for k in path[:-1]:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                ok = False
                break
        if ok and isinstance(node, dict):
            node[path[-1]] = value
            applied.append('.'.join(path))
    if applied:
        print(f"    set {what} via: {', '.join(applied)}")
    else:
        print(f"    !! could not locate a params path for {what} "
              f"(tried {path_options}) -- relying on the config file's own value")
    return applied


def _init_deepverse():
    """Build a DeepVerse Dataset with this script's antenna/OFDM settings."""
    try:
        import deepverse as dv
        from deepverse import ParameterManager
    except ImportError as e:
        raise RuntimeError(
            "DeepVerse is required to build the CSI cache but is not installed."
        ) from e

    cfg = CONFIG_PATH
    if not os.path.exists(cfg):
        for alt in ('config.m', 'config1.m'):
            alt_path = f"{SCENARIO_PATH}/param/{alt}"
            if os.path.exists(alt_path):
                cfg = alt_path
                break
    print(f"  Config file : {cfg}   (READ ONLY -- never modified)")

    pm = ParameterManager(cfg)

    pm.params["dataset_folder"] = str(PROJECT_ROOT / "scenarios")
    _yaml_check = PROJECT_ROOT / "scenarios" / SCENARIO_FOLDER / "param" / "scenario.yaml"
    assert _yaml_check.exists(), (
        f"scenario.yaml not found at {_yaml_check}. Check PROJECT_ROOT / SCENARIO_FOLDER.")

    pm.params["scenes"]          = list(range(TOTAL_SCENES))
    pm.params["basestations"]    = list(range(1, NUM_BS + 1))
    pm.params["radar"]["enable"] = False
    pm.params["comm"]["enable"]  = True
    pm.params["position"]        = True
    
    pm.params["camera"]          = True
    pm.params["lidar"]           = True
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

    print("  Overriding DeepVerse params in memory (config file untouched):")
    _set_nested(pm.params,
                [["comm", "bs_antenna", "shape"], ["comm", "bs_antenna_shape"]],
                list(ANT_SHAPE), f"bs_antenna.shape = {list(ANT_SHAPE)}")
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


def _verify_dv_shapes(dv_dataset, samples):
    """Pull one real sample and confirm the in-memory overrides actually took."""
    scene, bs, ue = samples[0]
    comm = dv_dataset.get_sample('comm-ue', index=scene, bs_idx=bs, ue_idx=ue)
    if comm is None or not hasattr(comm, 'coeffs') or comm.coeffs is None:
        raise RuntimeError(f"First sample {samples[0]} returned no comm coeffs.")
    got = int(np.asarray(comm.coeffs).size)
    want = NUM_ANT * NUM_SUB
    print(f"  Verifying CSI size: got {got} values, expected "
          f"{NUM_ANT} x {NUM_SUB} = {want}")
    if got != want:
        raise RuntimeError(
            f"DeepVerse returned {got} channel coefficients but this script is "
            f"configured for NUM_ANT={NUM_ANT} x NUM_SUB={NUM_SUB} = {want}.\n"
            f"The in-memory parameter override did not take effect (the params "
            f"nesting differs from every path tried in _set_nested).\n"
            f"FIX: print(pm.params) inside _init_deepverse to find the real key "
            f"path for bs_antenna.shape / OFDM.selected_subcarriers and add it "
            f"to the path list. Do NOT work around this by editing config.m if "
            f"that file is shared with other projects.")
    print("[OK] Overrides confirmed against real data")


def _scan_valid_samples():
    """{'samples','bs_positions','ue_positions'} -- reuses v6's file read-only
    when present, since validity/positions don't depend on antenna/subcarrier."""
    if not FORCE_RESCAN_SAMPLES:
        if OWN_SAMPLES_FILE.exists():
            with open(OWN_SAMPLES_FILE) as f:
                vdata = json.load(f)
            print(f"[OK] Loaded {len(vdata['samples']):,} samples from "
                  f"{OWN_SAMPLES_FILE.name}")
            return vdata
        if REUSE_V6_VALID_SAMPLES and V6_SAMPLES_FILE.exists():
            with open(V6_SAMPLES_FILE) as f:
                vdata = json.load(f)
            print(f"[OK] Reusing v6's {V6_SAMPLES_FILE.name} READ-ONLY "
                  f"({len(vdata['samples']):,} samples)")
            with open(OWN_SAMPLES_FILE, 'w') as f:
                json.dump(vdata, f)
            print(f"     copied to {OWN_SAMPLES_FILE.name} (v6's file untouched)")
            return vdata

    print("[..] Scanning DeepVerse for valid samples (one-time, slow)")
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
                    comm = dv_dataset.get_sample('comm-ue', index=scene,
                                                 bs_idx=bs, ue_idx=ue)
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
    with open(OWN_SAMPLES_FILE, 'w') as f:
        json.dump(vdata, f)
    print(f"[OK] Found {len(valid_samples):,} samples -> {OWN_SAMPLES_FILE.name}")
    return vdata


def _csi_cache_valid(samples) -> bool:
    """Checks row count, ant_shape, (num_ant, num_sub), and real dataset shape."""
    if not CSI_H5_FILE.exists():
        return False
    try:
        with h5py.File(CSI_H5_FILE, 'r') as f:
            if 'csi' not in f:
                return False
            shp = f['csi'].shape
            if shp[0] != len(samples):
                print(f"[..] cache has {shp[0]} rows, samples has {len(samples)}")
                return False
            if list(f.attrs.get('ant_shape', [])) != list(ANT_SHAPE):
                print(f"[..] cache ant_shape={list(f.attrs.get('ant_shape', []))} "
                      f"!= {list(ANT_SHAPE)}")
                return False
            if int(f.attrs.get('num_sub', -1)) != NUM_SUB or \
               int(f.attrs.get('num_ant', -1)) != NUM_ANT:
                print(f"[..] cache is {f.attrs.get('num_ant')}x{f.attrs.get('num_sub')}, "
                      f"need {NUM_ANT}x{NUM_SUB}")
                return False
            if tuple(shp[1:]) != (2, NUM_ANT, NUM_SUB):
                print(f"[..] cache dataset shape {shp} disagrees with its own attrs")
                return False
            if list(f.attrs.get('selected_subcarriers', [])) != list(SELECTED_SUBCARRIERS):
                print(f"[..] cache used a different subcarrier selection")
                return False
            if float(np.abs(f['csi'][:min(256, shp[0])]).std()) <= 0.0:
                print("[..] cache CSI is all-zero")
                return False
        return True
    except Exception as e:
        print(f"[..] cache unreadable ({e})")
        return False


def _build_csi_cache(samples):
    """Extracts CSI into our own HDF5. Writes to .tmp then moves into place
    so an aborted run can never leave a half-written cache."""
    dv_dataset = _init_deepverse()
    _verify_dv_shapes(dv_dataset, samples)

    N = len(samples)
    tmp = CSI_H5_FILE.with_suffix('.h5.tmp')
    if tmp.exists():
        tmp.unlink()
    print(f"[..] Building CSI cache: {N:,} samples -> {CSI_H5_FILE.name}")
    with h5py.File(tmp, 'w') as f:
        dset = f.create_dataset('csi', shape=(N, 2, NUM_ANT, NUM_SUB),
                                dtype='float32',
                                chunks=(min(500, N), 2, NUM_ANT, NUM_SUB))
        for idx in tqdm(range(N), desc="  csi (DeepVerse)"):
            scene, bs, ue = samples[idx]
            comm = dv_dataset.get_sample('comm-ue', index=scene, bs_idx=bs, ue_idx=ue)
            csi = np.asarray(comm.coeffs).reshape(NUM_ANT, NUM_SUB)
            dset[idx] = np.stack([csi.real, csi.imag], axis=0).astype(np.float32)
            if (idx + 1) % 1000 == 0:
                f.flush()
        f.attrs['ant_shape']            = list(ANT_SHAPE)
        f.attrs['num_ant']              = NUM_ANT
        f.attrs['num_sub']              = NUM_SUB
        f.attrs['selected_subcarriers'] = list(SELECTED_SUBCARRIERS)
        f.attrs['subcarrier_mode']      = SUBCARRIER_MODE
        f.attrs['scenario']             = SCENARIO_NAME
        f.attrs['built_by']             = 'FC_AE_GPR_Baseline_v1'
    shutil.move(str(tmp), str(CSI_H5_FILE))
    print(f"[OK] Wrote {CSI_H5_FILE.name}")


def load_deepverse_cache():
    """Ensures the exclusive CSI cache exists and is valid, loads it with
    the matching UE positions."""
    vdata = _scan_valid_samples()
    samples      = vdata['samples']
    ue_positions = vdata['ue_positions']
    bs_positions = vdata['bs_positions']

    if FORCE_REBUILD_CSI or not _csi_cache_valid(samples):
        if FORCE_REBUILD_CSI:
            print("[..] FORCE_REBUILD_CSI=True -- rebuilding")
        _build_csi_cache(samples)
    else:
        print(f"[OK] Reusing valid CSI cache {CSI_H5_FILE.name}")

    with h5py.File(CSI_H5_FILE, 'r') as f:
        csi = f['csi'][:]                                # (N, 2, NUM_ANT, NUM_SUB)

    # same clean-sample filter v6 applies, so both share a sample universe
    keep, loc = [], []
    for i, (scene, bs, ue) in enumerate(samples):
        uk, bk = f"{scene}_{ue}", f"{scene}_{bs}"
        if uk in ue_positions and bk in bs_positions:
            keep.append(i)
            loc.append(ue_positions[uk])
    keep = np.asarray(keep, dtype=np.int64)
    loc  = np.asarray(loc,  dtype=np.float64)            # (N, 3) metres

    csi = csi[keep]
    print(f"[OK] {len(keep):,} clean samples of {len(samples):,} "
          f"({100*len(keep)/max(1,len(samples)):.1f}%). csi={csi.shape}")
    return csi, loc



# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def dft_matrix_V(nt: int) -> np.ndarray:
    z = np.arange(nt).reshape(nt, 1)
    q = np.arange(nt).reshape(1, nt)
    return np.exp(-1j * 2 * np.pi * z * (q - nt / 2.0) / nt) / np.sqrt(nt)


def dft_matrix_F(nc: int) -> np.ndarray:
    z = np.arange(nc).reshape(nc, 1)
    q = np.arange(nc).reshape(1, nc)
    return np.exp(-1j * 2 * np.pi * z * q / nc) / np.sqrt(nc)


def csi_to_adp(csi: np.ndarray, adp_nt: int, adp_nc: int) -> np.ndarray:
    """(N, 2, n_ant, n_sub) raw Re/Im  ->  (N, adp_nt, adp_nc) ADP magnitude."""
    V = dft_matrix_V(adp_nt)
    F = dft_matrix_F(adp_nc)
    Vh = V.conj().T

    H = (csi[:, 0] + 1j * csi[:, 1]).astype(np.complex128)   # (N, n_ant, n_sub)
    H = H[:, :adp_nt, :adp_nc]
    A = np.abs(np.einsum('ij,njk,kl->nil', Vh, H, F))
    return A.astype(np.float32)


def build_or_load_adp(csi, loc):
    if ADP_CACHE_FILE.exists() and not FORCE_REBUILD_ADP:
        d = np.load(ADP_CACHE_FILE)
        cached = d['adp']
        if cached.shape[0] == csi.shape[0] and cached.shape[1:] == (ADP_NT, ADP_NC):
            print(f"[OK] Reusing ADP cache {ADP_CACHE_FILE.name}  {cached.shape}")
            return cached, d['loc']
        print(f"[..] ADP cache stale (has {cached.shape}, need "
              f"{(csi.shape[0], ADP_NT, ADP_NC)}); rebuilding")
    elif FORCE_REBUILD_ADP:
        print("[..] FORCE_REBUILD_ADP=True -- recomputing ADP")
    t0 = time.time()
    adp = csi_to_adp(csi, ADP_NT, ADP_NC)
    np.savez_compressed(ADP_CACHE_FILE, adp=adp, loc=loc)
    print(f"[OK] Built ADP {adp.shape} in {time.time()-t0:.1f}s -> {ADP_CACHE_FILE.name}")
    return adp, loc


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def augment_adp(adp_train: np.ndarray, max_samples: int = None,
                seed: int = RANDOM_SEED) -> np.ndarray:
    n, nt, nc = adp_train.shape
    step_r = max(1, nt // 8)
    step_c = max(1, nc // 8)
    shifts = [(r, c)
              for r in range(0, nt + 1, step_r)
              for c in range(0, nc + 1, step_c)]

    n_var = 6 if nt == nc else 4   # rot90 only preserves shape on a square ADP

    def variant(a, v):
        if v == 0:
            return a
        if nt == nc:
            if v == 1: return np.rot90(a, 1)
            if v == 2: return np.rot90(a, -1)
            if v == 3: return np.rot90(a, 2)
            if v == 4: return np.flipud(a)
            return np.fliplr(a)
        else:
            if v == 1: return np.rot90(a, 2)
            if v == 2: return np.flipud(a)
            return np.fliplr(a)

    total = n * n_var * len(shifts)
    if max_samples is not None and total > max_samples:
        # sample without materializing the full set
        rng = np.random.RandomState(seed)
        pick = rng.choice(total, size=max_samples, replace=False)
        pick.sort()
        out = np.empty((max_samples, nt, nc), dtype=np.float32)
        n_sv = n_var * len(shifts)
        for j, combo in enumerate(pick):
            i, rem = divmod(int(combo), n_sv)
            v, s = divmod(rem, len(shifts))
            r, c = shifts[s]
            out[j] = np.roll(np.roll(variant(adp_train[i], v), r, axis=0), c, axis=1)
        print(f"[OK] Augmented {n} ADPs -> {max_samples} of {total} possible "
              f"({len(shifts)} shifts x {n_var} variants; subsampled to cap memory)")
        return out

    chunks = []
    for i in range(n):
        for v in range(n_var):
            av = variant(adp_train[i], v)
            for (r, c) in shifts:
                chunks.append(np.roll(np.roll(av, r, axis=0), c, axis=1))
    aug = np.stack(chunks, axis=0).astype(np.float32)
    print(f"[OK] Augmented {n} ADPs -> {aug.shape[0]} "
          f"({len(shifts)} shifts x {n_var} variants)")
    return aug


# ---------------------------------------------------------------------------
# FC-AE (authors' FC_AE_Models.ipynb)
# dense-only enc/dec, LeakyReLU after every layer, MSE, Adam(5e-4), batch 32
# bottleneck is always 4x4=16
# ---------------------------------------------------------------------------
FCAE_LAYERS = {
    (64, 64): [1024, 256, 16],
    (32, 32): [256, 64, 16],
    (16, 16): [144, 64, 16],
    (8, 8):   [36, 16],
    (4, 4):   [16],
    (16, 8):  [72, 36, 16],   
}
ENCODED_DIM = 16


class FCAutoEncoder(nn.Module):
    def __init__(self, in_dim: int, widths: list, slope: float = LEAKY_SLOPE):
        super().__init__()
        enc, d = [], in_dim
        for w in widths:
            enc += [nn.Linear(d, w), nn.LeakyReLU(slope)]
            d = w
        self.encoder = nn.Sequential(*enc)

        dec = []
        for w in list(reversed(widths[:-1])) + [in_dim]:
            dec += [nn.Linear(d, w), nn.LeakyReLU(slope)]
            d = w
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def train_fcae(aug_adp: np.ndarray, verbose_every: int = 50):
    nt, nc = aug_adp.shape[1], aug_adp.shape[2]
    in_dim = nt * nc
    key = (nt, nc)
    if key not in FCAE_LAYERS:
        raise KeyError(f"No FC-AE architecture for ADP {nt}x{nc}. "
                       f"Known: {sorted(FCAE_LAYERS)}")
    widths = FCAE_LAYERS[key]
    if key == (16, 8):
        print("  !! ADP 16x8 uses INTERPOLATED FC-AE widths, not the authors' "
              "design -- results are not comparable to the paper's tables.")

    X = torch.from_numpy(aug_adp.reshape(-1, in_dim).astype(np.float32))
    n_val = int(round(FCAE_VAL_SPLIT * len(X)))
    X_tr, X_va = X[:len(X) - n_val], X[len(X) - n_val:]   # Keras validation_split takes the last fraction, no shuffle

    model = FCAutoEncoder(in_dim, widths).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=FCAE_LR)
    lossf = nn.MSELoss()
    dl = DataLoader(TensorDataset(X_tr), batch_size=FCAE_BATCH, shuffle=True)

    print(f"[..] FC-AE {in_dim} -> {' -> '.join(map(str, widths))} -> {in_dim} "
          f"| train {len(X_tr)}, val {len(X_va)}, {FCAE_EPOCHS} epochs")
    for ep in range(FCAE_EPOCHS):
        model.train()
        for (xb,) in dl:
            xb = xb.to(DEVICE)
            opt.zero_grad()
            recon, _ = model(xb)
            loss = lossf(recon, xb)
            loss.backward()
            opt.step()
        if verbose_every and ((ep + 1) % verbose_every == 0 or ep == 0):
            model.eval()
            with torch.no_grad():
                vb = X_va.to(DEVICE)
                vloss = lossf(model(vb)[0], vb).item() if len(X_va) else float('nan')
            # scientific notation on purpose -- a 1e-14 loss prints as 0.000000 in fixed point
            print(f"     epoch {ep+1:4d}/{FCAE_EPOCHS}  "
                  f"train {loss.item():.4e}  val {vloss:.4e}")
    # health check: did it actually learn anything
    model.eval()
    with torch.no_grad():
        probe = X[:min(512, len(X))].to(DEVICE)
        recon, z = model(probe)
        in_rms   = probe.pow(2).mean().sqrt().item()
        out_rms  = recon.pow(2).mean().sqrt().item()
        z_spread = z.std(dim=0).mean().item()
    print(f"     health: input RMS {in_rms:.3e}  recon RMS {out_rms:.3e}  "
          f"latent spread {z_spread:.3e}")
    if in_rms > 0 and out_rms < 0.01 * in_rms:
        raise RuntimeError(
            f"FC-AE collapsed: reconstruction RMS ({out_rms:.3e}) is under 1% of "
            f"the input RMS ({in_rms:.3e}), i.e. it learned to output ~zero.\n"
            f"This is the classic symptom of un-scaled inputs -- check that "
            f"ADP_SCALE_MODE is not 'none' and that the reported ADP scale "
            f"brought values to order 1.")
    if z_spread < 1e-6:
        raise RuntimeError(
            f"FC-AE latent code is constant across samples (spread {z_spread:.3e}); "
            f"the encoder carries no information and any downstream GPR would "
            f"just predict the mean.")
    return model


@torch.no_grad()
def encode_adp(model, adp: np.ndarray, batch: int = 4096) -> np.ndarray:
    model.eval()
    flat = adp.reshape(len(adp), -1).astype(np.float32)
    out = []
    for i in range(0, len(flat), batch):
        xb = torch.from_numpy(flat[i:i + batch]).to(DEVICE)
        out.append(model.encoder(xb).cpu().numpy())
    return np.concatenate(out, axis=0)


def fcae_similarity(model, adp: np.ndarray, batch: int = 4096) -> np.ndarray:
    """Paper Eq. (15): normalized correlation between input and reconstruction.
    The notebook's mse() helper actually computes <A,B>/(||A|| ||B||) -- a
    similarity, despite the name."""
    model.eval()
    flat = adp.reshape(len(adp), -1).astype(np.float32)
    sims = []
    with torch.no_grad():
        for i in range(0, len(flat), batch):
            xb = torch.from_numpy(flat[i:i + batch]).to(DEVICE)
            rb, _ = model(xb)
            a, b = xb, rb
            na = torch.linalg.norm(a, dim=1)
            nb = torch.linalg.norm(b, dim=1)
            s = (a * b).sum(dim=1) / torch.clamp(na * nb, min=1e-12)
            s[(na == 0) | (nb == 0)] = 0.0
            sims.append(s.cpu().numpy())
    return np.concatenate(sims)


# ---------------------------------------------------------------------------
# GPR (authors' GPR_train_kfold.m / GPR_test_kfold.m)
# two independent single-output GPRs (x, y) instead of one multi-output,
# per paper Sec. IV-B
# ---------------------------------------------------------------------------
def _median_lengthscale(X, max_n=400, seed=RANDOM_SEED):
    """Median pairwise distance heuristic for the GPR init length scale.

    The authors' l=0.01 only makes sense at their ADP's own scale -- on our
    rescaled inputs (typical pairwise distance ~24) it underflows every
    off-diagonal kernel entry to zero, K becomes the identity, and the GP
    just returns its prior mean everywhere (flat log-marginal-likelihood,
    L-BFGS can't climb out). The median heuristic sets l to the data's own
    distance scale so the optimizer starts somewhere useful.
    """
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    if n < 2:
        return 1.0
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, min(n, max_n), replace=False)
    Z = X[idx]
    sq = (Z ** 2).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
    d = np.sqrt(np.clip(d2, 0.0, None))
    d = d[np.triu_indices_from(d, k=1)]
    d = d[d > 0]
    return float(np.median(d)) if d.size else 1.0


def make_gpr(length_scale=None):
    ls = GPR_LENGTH_SCALE if length_scale is None else length_scale
   
    lo, hi = ls / GPR_LS_BOUND_FACTOR, ls * GPR_LS_BOUND_FACTOR
    kernel = (ConstantKernel(GPR_SIGMA_F ** 2, (1e-3, 1e12))
              * Matern(length_scale=ls, length_scale_bounds=(lo, hi), nu=1.5)
              + WhiteKernel(GPR_SIGMA_N ** 2, (1e-12, 1e4)))
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                    n_restarts_optimizer=GPR_RESTARTS,
                                    random_state=RANDOM_SEED)


def run_gpr(train_X, train_Y, test_X, tag=''):
    """train_Y is (n,2). Returns (m,2) predictions from two single-output GPRs."""
    ls = (_median_lengthscale(train_X) if GPR_MEDIAN_HEURISTIC
          else GPR_LENGTH_SCALE)
    preds = []
    for axis in range(2):
        g = make_gpr(ls)
        g.fit(train_X, train_Y[:, axis])
        preds.append(g.predict(test_X))
    pred = np.stack(preds, axis=1)

    spread = float(np.std(pred, axis=0).mean())
    tgt_spread = float(np.std(train_Y, axis=0).mean())
    if tgt_spread > 0 and spread < 1e-3 * tgt_spread:
        print(f"     !! GPR{(' ' + tag) if tag else ''} collapsed to a constant "
              f"(pred spread {spread:.3e} vs target spread {tgt_spread:.3e}); "
              f"init length_scale={ls:.3e}")
    return pred


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
class SamePad2d(nn.Module):
    def __init__(self, k):
        super().__init__()
        lo = (k - 1) // 2
        hi = (k - 1) - lo
        self.pad = nn.ZeroPad2d((lo, hi, lo, hi))

    def forward(self, x):
        return self.pad(x)


def _conv_block(cin, cout, k, bn=True):
    layers = [SamePad2d(k), nn.Conv2d(cin, cout, k), nn.ReLU()]
    if bn:
        layers.append(nn.BatchNorm2d(cout))
    return layers
CNN_SPECS = {
    4:  [(32, 5, True, True), (64, 3, True, True)],
    8:  [(16, 7, True, True), (32, 5, True, False), (32, 5, True, True),
         (64, 3, True, True)],
    16: [(8, 8, True, True), (16, 7, True, True), (32, 5, True, False),
         (32, 5, True, True), (64, 3, True, True)],
    32: [(4, 16, True, True), (8, 8, True, True), (16, 7, True, True),
         (32, 5, True, False), (32, 5, True, True), (64, 3, True, True)],
    64: [(2, 32, True, True), (4, 16, True, True), (8, 8, False, True),
         (16, 7, True, True), (32, 5, True, True), (64, 3, True, True)],
}


class ADPCnn(nn.Module):
    def __init__(self, nt, nc, out_dim=3):
        super().__init__()
        side = max(nt, nc)
        if side not in CNN_SPECS:
            raise KeyError(f"No CNN spec for ADP side {side}. Known: {sorted(CNN_SPECS)}")
        blocks, cin, h, w = [], 1, nt, nc
        for (cout, k, bn, pool) in CNN_SPECS[side]:
            blocks += _conv_block(cin, cout, k, bn)
            cin = cout
            if pool:
               
                ph = 2 if h > 1 else 1
                pw = 2 if w > 1 else 1
                if ph > 1 or pw > 1:
                    blocks.append(nn.MaxPool2d((ph, pw)))
                    h, w = h // ph, w // pw
        self.features = nn.Sequential(*blocks)
        self.head = nn.Linear(cin * h * w, out_dim)

    def forward(self, x):
        return self.head(torch.flatten(self.features(x), 1))


def train_cnn(train_adp, train_loc, epochs=CNN_EPOCHS, verbose_every=200):
    n, nt, nc = train_adp.shape
    X = torch.from_numpy(train_adp[:, None].astype(np.float32))
    Yraw = np.asarray(train_loc, dtype=np.float64)

    # standardize targets from this fold's training split only -- no test leak
    if CNN_NORMALIZE_TARGETS:
        y_mean = Yraw.mean(axis=0)
        y_std  = Yraw.std(axis=0)
        y_std[y_std < 1e-8] = 1.0
    else:
        y_mean = np.zeros(Yraw.shape[1])
        y_std  = np.ones(Yraw.shape[1])
    Y = torch.from_numpy(((Yraw - y_mean) / y_std).astype(np.float32))

    n_val = max(1, int(round(CNN_VAL_SPLIT * n)))
    X_tr, Y_tr = X[:n - n_val], Y[:n - n_val]
    X_va, Y_va = X[n - n_val:], Y[n - n_val:]

    model = ADPCnn(nt, nc, out_dim=Y.shape[1]).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters())
    lossf = nn.L1Loss()
    dl = DataLoader(TensorDataset(X_tr, Y_tr),
                    batch_size=min(CNN_BATCH, max(1, len(X_tr))), shuffle=True)

    best, best_state, since = float('inf'), None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = lossf(model(X_va.to(DEVICE)), Y_va.to(DEVICE)).item()
        if v < best - 1e-9:
            best, since = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            since += 1
            if since >= CNN_PATIENCE:
                break
        if verbose_every and (ep + 1) % verbose_every == 0:
            print(f"     cnn epoch {ep+1}/{epochs}  val_mae {v:.4f}  best {best:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    model._y_mean = y_mean
    model._y_std  = y_std
    return model


@torch.no_grad()
def predict_cnn(model, adp, batch=512):
    model.eval()
    X = torch.from_numpy(adp[:, None].astype(np.float32))
    out = [model(X[i:i + batch].to(DEVICE)).cpu().numpy()
           for i in range(0, len(X), batch)]
    pred = np.concatenate(out, axis=0).astype(np.float64)
    return pred * getattr(model, '_y_std', 1.0) + getattr(model, '_y_mean', 0.0)



# ---------------------------------------------------------------------------
def correlation_predict(train_adp, train_loc, test_adp, block=512):
    Tr = train_adp.reshape(len(train_adp), -1).astype(np.float64)
    Te = test_adp.reshape(len(test_adp), -1).astype(np.float64)
    Tr /= np.clip(np.linalg.norm(Tr, axis=1, keepdims=True), 1e-12, None)
    Te /= np.clip(np.linalg.norm(Te, axis=1, keepdims=True), 1e-12, None)
    preds = []
    for i in range(0, len(Te), block):
        preds.append(train_loc[np.argmax(Te[i:i + block] @ Tr.T, axis=1)])
    return np.concatenate(preds, axis=0)


# ---------------------------------------------------------------------------
# k-fold evaluation
# ---------------------------------------------------------------------------
def rmse_immse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def fold_indices(total, size, seed, absolute=False):
    """size is a percentage (absolute=False) or a label count (absolute=True)."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(total)
    n_train = int(size) if absolute else int(round(total * size / 100.0))
    start   = int(round(total * TEST_FRACTION))
    assert n_train >= 2, f"n_train={n_train} is too small to fit any model"
    assert n_train <= start, (
        f"n_train={n_train} would overlap the test block, which starts at "
        f"index {start}. With TEST_FRACTION={TEST_FRACTION} the largest "
        f"leak-free training set is {start} samples.")
    train_idx = perm[:n_train]
    test_idx  = perm[start:start + TEST_SIZE]
    assert len(np.intersect1d(train_idx, test_idx)) == 0, "train/test overlap"
    return train_idx, test_idx


def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    csi, loc = load_deepverse_cache()
    adp, loc = build_or_load_adp(csi, loc)
    del csi
    loc2 = loc[:, :2]                       
    total = len(adp)

    # global ADP scale, estimated from the unlabeled slice only
    rng = np.random.RandomState(RANDOM_SEED)
    perm = rng.permutation(total)
    if AUG_N_SOURCE is not None:
        n_aug_src = min(int(AUG_N_SOURCE), total)
        _src_note = f"absolute AUG_N_SOURCE={AUG_N_SOURCE}"
    else:
        n_aug_src = max(1, int(round(total * AUG_PERCENT)))
        _src_note = f"AUG_PERCENT={AUG_PERCENT*100:.2f}%"
    aug_src_raw = adp[perm[:n_aug_src]]

    raw_rms = float(np.sqrt(np.mean(aug_src_raw.astype(np.float64) ** 2)))
    if ADP_SCALE_MODE == 'std':
        ADP_SCALE = 1.0 / max(float(aug_src_raw.std()), 1e-30)
    elif ADP_SCALE_MODE == 'max':
        ADP_SCALE = 1.0 / max(float(aug_src_raw.max()), 1e-30)
    elif ADP_SCALE_MODE == 'none':
        ADP_SCALE = 1.0
    else:
        raise ValueError("ADP_SCALE_MODE must be 'std', 'max' or 'none'")

    print(f"\n[..] ADP raw RMS = {raw_rms:.3e}  ->  scale = {ADP_SCALE:.3e} "
          f"(mode='{ADP_SCALE_MODE}', from the {n_aug_src}-sample unlabeled slice)")
    if ADP_SCALE_MODE == 'none' and raw_rms < 1e-3:
        print("  !! ADP_SCALE_MODE='none' with tiny ADP values -- the FC-AE and CNN "
              "will almost certainly collapse. The health check will stop the run.")
    adp = (adp * ADP_SCALE).astype(np.float32)
    print(f"[OK] Scaled ADP RMS = {np.sqrt(np.mean(adp.astype(np.float64)**2)):.4f} "
          f"(order 1 => MSE/Adam behave normally)")
    aug_src = adp[perm[:n_aug_src]]

    # FC-AE: trained once on an augmented slice of unlabeled data
    eadp, sim = None, None
    if RUN_FCAE:
        print(f"\n[..] FC-AE source: {n_aug_src} unique ADPs ({_src_note}); "
              f"{100*n_aug_src/total:.2f}% of the dataset")
        if n_aug_src / total > 0.25:
            print(f"  !! That is a large share of the dataset. The FC-AE is "
                  f"UNSUPERVISED (labels are never used), so this is not label "
                  f"leakage, but state it explicitly when reporting.")
        aug = augment_adp(aug_src, max_samples=AUG_MAX_SAMPLES)
        fcae = train_fcae(aug)
        del aug

        sim = fcae_similarity(fcae, adp)
        print(f"[OK] FC-AE reconstruction similarity (paper Eq. 15): "
              f"{sim.mean():.4f} +/- {sim.std():.4f}   (paper Table V: ~0.83-0.92)")

        eadp = encode_adp(fcae, adp)
        print(f"[OK] Encoded ADP: {eadp.shape}  (bottleneck {ENCODED_DIM} = 4x4)")
    else:
        print(f"\n[--] FC-AE skipped: ADP is already {ADP_NT}x{ADP_NC} "
              f"(= the 4x4 bottleneck), so there is nothing to compress. "
              f"The paper's 4x4 block likewise has no FC-AE-GPR row.")

    # k-fold over train fractions
    methods = ['CNN', 'Correlation']
    if RUN_FCAE:
        methods.insert(0, 'FC-AE-GPR')
    if RUN_RAW_GPR:
        methods.insert(1 if RUN_FCAE else 0, 'GPR')
    else:
        print(f"[--] Raw-ADP GPR skipped: {ADP_NT*ADP_NC}-dim input exceeds "
              f"RAW_GPR_MAX_DIM={RAW_GPR_MAX_DIM}. The authors likewise report "
              f"no plain-GPR row for 32x32 or 64x64.")
    print(f"[OK] Methods this run: {methods}")

    ABSOLUTE = (TRAIN_SIZE_MODE == 'absolute')
    SWEEP = TRAIN_SIZES_ABS if ABSOLUTE else PERCENT_ALL
    print(f"[OK] Train-size mode: '{TRAIN_SIZE_MODE}' -> {SWEEP}")

    results = {m: {p: [] for p in SWEEP} for m in methods}
    flat = adp.reshape(total, -1)          # raw-ADP view for the no-FC-AE GPR

    for pct in SWEEP:
        for fold in range(1, N_FOLDS + 1):
            tr, te = fold_indices(total, pct, seed=fold, absolute=ABSOLUTE)
            t0 = time.time()

            if 'FC-AE-GPR' in results:
                results['FC-AE-GPR'][pct].append(
                    rmse_immse(loc2[te], run_gpr(eadp[tr], loc2[tr], eadp[te],
                                                 tag='(FC-AE-GPR)')))

            if 'GPR' in results:
                results['GPR'][pct].append(
                    rmse_immse(loc2[te], run_gpr(flat[tr], loc2[tr], flat[te],
                                                 tag='(raw ADP)')))

            cnn = train_cnn(adp[tr], loc[tr][:, :3], verbose_every=0)
            results['CNN'][pct].append(
                rmse_immse(loc2[te], predict_cnn(cnn, adp[te])[:, :2]))
            del cnn

            results['Correlation'][pct].append(
                rmse_immse(loc2[te], correlation_predict(adp[tr], loc2[tr], adp[te])))

            _lbl = f"n={pct:>5}" if ABSOLUTE else f"pct={pct:>5}%"
            print(f"  {_lbl}  fold {fold}/{N_FOLDS}  "
                  f"n_train={len(tr):>5}  ({time.time()-t0:.1f}s)  "
                  + "  ".join(f"{m}={results[m][pct][-1]:.3f}" for m in methods))

    # report
    print("\n" + "=" * 72)
    print(f"  RMSE [m] over {N_FOLDS} folds -- {EXP_NAME}")
    print(f"  ADP {ADP_NT}x{ADP_NC}, test block = {TEST_SIZE} samples")
    print("=" * 72)
    _unit = "" if ABSOLUTE else "%"
    hdr = f"{'Method':<14}" + "".join(f"{p:>10}{_unit}" for p in SWEEP)
    print(hdr)
    print("-" * len(hdr))
    for m in methods:
        row = f"{m:<14}"
        for p in SWEEP:
            v = results[m][p]
            row += f"{np.mean(v):>10.3f} " if v else f"{'--':>10} "
        print(row)
    print("-" * len(hdr))
    for m in methods:
        row = f"{m + ' (std)':<14}"
        for p in SWEEP:
            v = results[m][p]
            row += f"{np.std(v):>10.3f} " if v else f"{'--':>10} "
        print(row)

    out = RESULTS_DIR / f"{EXP_NAME}_results.json"
    with open(out, 'w') as f:
        json.dump({
            'experiment': EXP_NAME,
            'scenario': SCENARIO_NAME,
            'ant_shape': list(ANT_SHAPE),
            'adp_preset': ADP_PRESET,
            'adp_shape': [ADP_NT, ADP_NC],
            'n_folds': N_FOLDS,
            'train_size_mode': TRAIN_SIZE_MODE,
            'sweep': list(SWEEP),
            'total_samples': int(total),
            'test_size': TEST_SIZE,
            'fcae_similarity_mean': (float(sim.mean()) if sim is not None else None),
            'fcae_similarity_std': (float(sim.std()) if sim is not None else None),
            'ran_fcae': bool(RUN_FCAE),
            'ran_raw_gpr': bool(RUN_RAW_GPR),
            'subcarrier_mode': SUBCARRIER_MODE,
            'selected_subcarriers': list(SELECTED_SUBCARRIERS),
            'delay_resolution_m': float(DELAY_RES_M),
            'unambiguous_range_m': float(UNAMBIG_M),
            'rmse_metric': 'immse (mean over all elements of the (n,2) residual, then sqrt)',
            'results': {m: {str(p): results[m][p] for p in SWEEP} for m in methods},
        }, f, indent=2)
    print(f"\n[OK] Saved -> {out}")


if __name__ == '__main__':
    main()
