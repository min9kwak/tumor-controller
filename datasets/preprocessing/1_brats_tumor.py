import os
import nibabel as nib
import numpy as np
import pickle
import tqdm
from monai.transforms import SpatialPad

# Set directory
brats_dir = 'D:/data/BraTS/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
out_dir = 'D:/data/tumor-controller'
tumor_out_dir = os.path.join(out_dir, "tumor")
normal_out_dir = os.path.join(out_dir, "normal")
os.makedirs(tumor_out_dir, exist_ok=True)
os.makedirs(normal_out_dir, exist_ok=True)

# Tumor criteria
tumor_lower = 1000
tumor_upper = 3000

# Min-Max Scaling
def normalize_volume(vol):
    vol_min, vol_max = np.min(vol), np.max(vol)
    return (vol - vol_min) / (vol_max - vol_min + 1e-8)  # prevent div by 0

# Z-slice
slice_range = list(range(80, 130))  # 155 total slices

# Padding Function
padding_function = SpatialPad(spatial_size=(256, 256), method='symmetric', mode='edge')

# Load patient directories
patient_dirs = next(os.walk(brats_dir))[1]
patient_dirs = [os.path.join(brats_dir, f) for f in patient_dirs]

for patient_dir in tqdm.tqdm(patient_dirs):
    patient_id = os.path.basename(patient_dir)

    filenames = [os.path.join(patient_dir, f) for f in os.listdir(patient_dir)]
    t1ce_filename = [f for f in filenames if '_t1ce.nii' in f][0]
    t2_filename = [f for f in filenames if '_t2.nii' in f][0]
    seg_filename = [f for f in filenames if '_seg.nii' in f][0]

    # Load and transpose
    t1ce = nib.load(t1ce_filename).get_fdata().transpose(2, 0, 1)  # (Z, H, W)
    t2 = nib.load(t2_filename).get_fdata().transpose(2, 0, 1)
    seg = nib.load(seg_filename).get_fdata().transpose(2, 0, 1)

    t1ce = normalize_volume(t1ce)
    t2 = normalize_volume(t2)

    for z in slice_range:
        # Padding and Shaping: (1, H, W)
        t1ce_slice = padding_function(t1ce[[z]]).numpy()
        t2_slice = padding_function(t2[[z]]).numpy()
        seg_slice = padding_function(seg[[z]]).numpy()[0]  # (H, W)

        # Integrate Tumor Mask
        tumor_mask = np.isin(seg_slice, [1, 2, 4]).astype(np.uint8)
        tumor_pixel_count = np.sum(tumor_mask)

        prefix = f'{patient_id}-z{z}'

        if tumor_pixel_count == 0:
            # Save Normal
            with open(os.path.join(normal_out_dir, f'{prefix}-t1ce.pkl'), 'wb') as f:
                pickle.dump(t1ce_slice, f)
            with open(os.path.join(normal_out_dir, f'{prefix}-t2.pkl'), 'wb') as f:
                pickle.dump(t2_slice, f)

        elif tumor_lower <= tumor_pixel_count <= tumor_upper:
            # Save Tumor
            with open(os.path.join(tumor_out_dir, f'{prefix}-t1ce.pkl'), 'wb') as f:
                pickle.dump(t1ce_slice, f)
            with open(os.path.join(tumor_out_dir, f'{prefix}-t2.pkl'), 'wb') as f:
                pickle.dump(t2_slice, f)
            with open(os.path.join(tumor_out_dir, f'{prefix}-seg.pkl'), 'wb') as f:
                pickle.dump(tumor_mask[np.newaxis, ...], f)  # (1, H, W)
