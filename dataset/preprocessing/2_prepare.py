import os
import json
import pickle
import tqdm
import cv2
import nibabel as nib
import numpy as np
import pandas as pd
from monai.transforms import SpatialPad
import sys

from utils.util import set_env

# set directory
config = dict(server='psc')
config = set_env(config)

data_root = config['data_root']
tumor_dir = os.path.join(data_root, 'tumor')
healthy_dir = os.path.join(data_root, 'healthy')

# Check if data_root exists
if not os.path.exists(data_root):
    print(f"Error: data_root directory does not exist: {data_root}")
    sys.exit(1)

# Check if required CSV files exist
survival_info_path = os.path.join(data_root, 'survival_info.csv')
tumor_stats_path = os.path.join(data_root, 'tumor_slice_stats.csv')

missing_files = []
if not os.path.exists(survival_info_path):
    missing_files.append(survival_info_path)
if not os.path.exists(tumor_stats_path):
    missing_files.append(tumor_stats_path)

if missing_files:
    print("Error: Required files do not exist:")
    for f in missing_files:
        print(f"  - {f}")
    sys.exit(1)

# prepare demo info and tumor info
demo_info = pd.read_csv(survival_info_path)
tumor_info = pd.read_csv(tumor_stats_path, dtype={'patient_id': str})
demo_info['patient_id'] = demo_info['Brats20ID'].str.split('_').str[-1]

# set directory
brats_dir = os.path.join(os.path.dirname(data_root), 'BraTS', 'MICCAI_BraTS2020_TrainingData', 'MICCAI_BraTS2020_TrainingData')

# Check if BraTS directory exists
if not os.path.exists(brats_dir):
    print(f"Error: BraTS data directory does not exist: {brats_dir}")
    sys.exit(1)

# Check if BraTS directory contains any data
if not any(os.path.isdir(os.path.join(brats_dir, d)) for d in os.listdir(brats_dir)):
    print(f"Error: No patient directories found in BraTS directory: {brats_dir}")
    sys.exit(1)

out_dir = data_root
tumor_out_dir = os.path.join(out_dir, "tumor")
normal_out_dir = os.path.join(out_dir, "healthy")

# Create output directories
os.makedirs(tumor_out_dir, exist_ok=True)
os.makedirs(normal_out_dir, exist_ok=True)

# hyperparameters
param = {
    "tumor_lower": 1000,
    "tumor_upper": 3000,
    "slice_range": list(range(80, 130)),
    "spatial_size": (256, 256),
    "canny_thresholds": {"low": 100, "high": 200},
    "normalization": {
        "type": "percentile_minmax",
        "percentile": 99.5
    }
}
with open(os.path.join(out_dir, "preprocessing_config.json"), "w") as f:
    json.dump(param, f, indent=4)

# set hyperparameters
slice_range  = param["slice_range"]
tumor_lower  = param["tumor_lower"]
tumor_upper  = param["tumor_upper"]
low_thresh   = param["canny_thresholds"]["low"]
high_thresh  = param["canny_thresholds"]["high"]
spatial_size = tuple(param["spatial_size"])

padding_function = SpatialPad(
    spatial_size=(-1, spatial_size[0], spatial_size[1]),
    method='symmetric', mode='edge'
)

# util functions
def normalize_volume_01(vol):
    """
    Normalize volume by:
    1. Clipping at 99.5 percentile (only for non-zero values)
    2. Min-max normalization to [0,1]
    """
    # Calculate 99.5 percentile on non-zero values
    non_zero_mask = vol != 0
    if non_zero_mask.any():
        p995 = np.percentile(vol[non_zero_mask], 99.5)
        vol = np.clip(vol, a_min=0, a_max=p995)
    
    # Min-max normalization
    v_min, v_max = vol.min(), vol.max()
    vol_01 = (vol - v_min) / (v_max - v_min + 1e-8)
    return vol_01, v_min, v_max

def generate_canny(vol_slice_01):
    """0~1 float → uint8 0~255 → Canny → binary uint8"""
    vol_uint8 = (vol_slice_01 * 255).astype(np.uint8)
    edges = cv2.Canny(vol_uint8, low_thresh, high_thresh)
    return (edges > 0).astype(np.uint8)

stat_records = []
tumor_slice_records = []  # New list for tumor slice statistics

# main loop
patient_dirs = [os.path.join(brats_dir, d)
                for d in next(os.walk(brats_dir))[1]]

for patient_dir in tqdm.tqdm(patient_dirs, desc="Patients"):
    patient_id = os.path.basename(patient_dir).split('_')[-1]

    t1ce_path = next(f for f in os.listdir(patient_dir) if '_t1ce.nii' in f)
    t2_path   = next(f for f in os.listdir(patient_dir) if '_t2.nii'   in f)
    seg_path  = next(f for f in os.listdir(patient_dir) if '_seg.nii'  in f)

    t1ce_raw = nib.load(os.path.join(patient_dir, t1ce_path)).get_fdata().transpose(2,0,1)
    t2_raw   = nib.load(os.path.join(patient_dir, t2_path)).get_fdata().transpose(2,0,1)
    seg      = nib.load(os.path.join(patient_dir, seg_path)).get_fdata().transpose(2,0,1)

    # Apply padding first
    t1ce_padded = padding_function(t1ce_raw[np.newaxis]).numpy()[0]  # (155, 256, 256)
    t2_padded = padding_function(t2_raw[np.newaxis]).numpy()[0]      # (155, 256, 256)
    seg_padded = padding_function(seg[np.newaxis]).numpy()[0]       # (155, 256, 256)
    
    # Normalize padded volumes
    t1ce_01, t1_min, t1_max = normalize_volume_01(t1ce_padded)
    t2_01,  t2_min, t2_max  = normalize_volume_01(t2_padded)

    stat_records += [
        {"patient_id": patient_id, "modality": "t1ce", "min": t1_min, "max": t1_max},
        {"patient_id": patient_id, "modality": "t2",   "min": t2_min, "max": t2_max},
    ]

    # ---- (b) for each slice ------------------------
    for z in slice_range:
        t1ce_slice = t1ce_01[z][np.newaxis]  # (1, 256, 256)
        t2_slice = t2_01[z][np.newaxis]      # (1, 256, 256)
        seg_slice = seg_padded[z]            # (256, 256)
        
        tumor_mask = np.isin(seg_slice, [1, 2, 4]).astype(np.uint8)
        tumor_pixels = tumor_mask.sum()

        # Edge map: generate from normalized padded slices
        canny_t1ce = generate_canny(t1ce_01[z])
        canny_t2   = generate_canny(t2_01[z])
        canny_t1ce = canny_t1ce[np.newaxis]  # (1, 256, 256)
        canny_t2   = canny_t2[np.newaxis]    # (1, 256, 256)

        # set saving directory
        if tumor_pixels == 0:
            prefix = os.path.join(normal_out_dir, f"{patient_id}")
        elif tumor_lower <= tumor_pixels <= tumor_upper:
            prefix = os.path.join(tumor_out_dir, f"{patient_id}")
            # Record tumor slice statistics
            tumor_slice_records.append({
                "patient_id": patient_id,
                "slice_num": z,
                "tumor_pixels": int(tumor_pixels)
            })
        else:
            continue

        # ---- (c) save(.pkl) -------------------------
        with open(f"{prefix}-brain-t1ce-{z}.pkl", "wb") as f:
            pickle.dump(t1ce_slice, f)
        with open(f"{prefix}-brain-t2-{z}.pkl", "wb") as f:
            pickle.dump(t2_slice, f)

        with open(f"{prefix}-edge-t1ce-{z}.pkl", "wb") as f:
            pickle.dump(canny_t1ce, f)
        with open(f"{prefix}-edge-t2-{z}.pkl", "wb") as f:
            pickle.dump(canny_t2, f)

        if tumor_pixels > 0:
            with open(f"{prefix}-seg-{z}.pkl", "wb") as f:
                pickle.dump(tumor_mask[np.newaxis], f)

# --------------------------------------------------
# 4. Save Stats CSV
# --------------------------------------------------
pd.DataFrame(stat_records).to_csv(
    os.path.join(out_dir, "normalization_minmax_stats.csv"), index=False
)

# Save tumor slice statistics
pd.DataFrame(tumor_slice_records).to_csv(
    os.path.join(out_dir, "tumor_slice_stats.csv"), index=False
)
