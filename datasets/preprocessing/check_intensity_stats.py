import os
import numpy as np
import nibabel as nib
import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_image_statistics(image_data, modality_name):
    """
    Analyze intensity statistics of a medical image
    """
    # Remove zero values
    non_zero_data = image_data[image_data != 0]
    
    # Basic statistics
    stats = {
        'min': np.min(non_zero_data),
        'max': np.max(non_zero_data),
        'mean': np.mean(non_zero_data),
        'std': np.std(non_zero_data),
        'median': np.median(non_zero_data),
    }
    
    # Percentiles
    percentiles = [95, 97, 99, 99.9]
    for p in percentiles:
        stats[f'{p}th_percentile'] = np.percentile(non_zero_data, p)
    
    print(f"\n{modality_name} Statistics (excluding zero values):")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")
    
    return stats

def plot_intensity_distribution(image_data, modality_name, stats, save_path=None):
    """
    Plot intensity distribution with marked statistics
    """
    plt.figure(figsize=(12, 6))
    
    # Plot histogram of non-zero values
    non_zero_data = image_data[image_data != 0]
    plt.hist(non_zero_data.flatten(), bins=100, density=True, alpha=0.7)
    
    # Plot vertical lines for important statistics
    plt.axvline(stats['mean'], color='r', linestyle='--', label='Mean')
    plt.axvline(stats['99.9th_percentile'], color='purple', linestyle='--', label='99.9th percentile')
    plt.axvline(stats['99th_percentile'], color='g', linestyle='--', label='99th percentile')
    plt.axvline(stats['95th_percentile'], color='b', linestyle='--', label='95th percentile')
    
    plt.title(f'Intensity Distribution (Non-zero values) - {modality_name}')
    plt.xlabel('Intensity Value')
    plt.ylabel('Density')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def main():
    # Get the base directory containing patient folders
    brats_dir = Path('D:/data/BraTS/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData')
    patient_dirs = [d for d in brats_dir.iterdir() if d.is_dir()]
    
    all_stats_t1ce = []
    all_stats_t2 = []
    
    for patient_dir in tqdm.tqdm(patient_dirs[:20], desc="Analyzing patients"):
        patient_id = patient_dir.name.split('_')[-1]
        
        # Load images
        t1ce_path = next(f for f in os.listdir(patient_dir) if '_t1ce.nii' in f)
        t2_path = next(f for f in os.listdir(patient_dir) if '_t2.nii' in f)
        
        t1ce_raw = nib.load(os.path.join(patient_dir, t1ce_path)).get_fdata().transpose(2,0,1)
        t2_raw = nib.load(os.path.join(patient_dir, t2_path)).get_fdata().transpose(2,0,1)
        
        # Analyze statistics
        stats_t1ce = analyze_image_statistics(t1ce_raw, f"T1CE - Patient {patient_id}")
        stats_t2 = analyze_image_statistics(t2_raw, f"T2 - Patient {patient_id}")
        
        all_stats_t1ce.append(stats_t1ce)
        all_stats_t2.append(stats_t2)
        
        # Plot distributions
        os.makedirs('intensity_plots', exist_ok=True)
        plot_intensity_distribution(
            t1ce_raw, 
            f"T1CE - Patient {patient_id}", 
            stats_t1ce,
            f'intensity_plots/patient_{patient_id}_t1ce.png'
        )
        plot_intensity_distribution(
            t2_raw, 
            f"T2 - Patient {patient_id}", 
            stats_t2,
            f'intensity_plots/patient_{patient_id}_t2.png'
        )

if __name__ == "__main__":
    main()