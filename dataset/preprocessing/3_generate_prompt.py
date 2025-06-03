import os
import pandas as pd
import json
import tqdm
from utils.util import set_env

config = {
    'server': 'psc',
}

# function
def classify_tumor_size(pixels: int) -> str:
    if pixels <= 1400:
        return "small tumor"
    elif pixels <= 1700:
        return "mild tumor"
    elif pixels <= 2000:
        return "medium-sized tumor"
    elif pixels <= 2300:
        return "moderate tumor"
    else:
        return "large tumor"

def generate_prompt(modality: str, age: int | None, tumor_pixels: int, force_healthy: bool = False) -> str:
    modality_str = {
        "t1ce": "A contrast-enhanced T1-weighted brain MRI",
        "t2": "A T2-weighted brain MRI"
    }.get(modality.lower(), "A brain MRI")

    age_desc = f"{age}-year-old" if age is not None else "unknown age"

    if force_healthy or tumor_pixels == 0:
        subject_desc = f"a {age_desc} healthy individual"
    else:
        size_desc = classify_tumor_size(tumor_pixels)
        subject_desc = f"a {age_desc} patient with a {size_desc}"

    return f"{modality_str} of {subject_desc}."


# set config
config = set_env(config)

# save metadata
meta_dir = os.path.join(config['data_root'], 'meta')

tumor_stats = os.path.join(config['data_root'], 'tumor_slice_stats.csv')
survival_info = os.path.join(config['data_root'], 'survival_info.csv')

tumor_stats = pd.read_csv(tumor_stats, dtype={'patient_id': str})
survival_info = pd.read_csv(survival_info)

survival_info['patient_id'] = survival_info['Brats20ID'].str.split('_').str[-1]

# tumor
tumor_dir = os.path.join(config['data_root'], 'tumor')
tumor_files = [os.path.join(tumor_dir, f) for f in os.listdir(tumor_dir) if '-brain-' in f]
tumor_meta_dir = os.path.join(meta_dir, 'tumor')
os.makedirs(tumor_meta_dir, exist_ok=True)

for tumor_file in tqdm.tqdm(tumor_files):
    patient_id, _, modality, slice = tumor_file.split('/')[-1].split('-')
    slice = slice.replace('.pkl', '')

    tumor_size = str(tumor_stats[tumor_stats['patient_id'] == patient_id]['tumor_pixels'].iloc[0])
    
    try:
        age = int(survival_info[survival_info['patient_id'] == patient_id]['Age'].iloc[0])
    except (IndexError, ValueError):
        age = None
    
    # generate prompt: modality, tumor size, age
    prompt = generate_prompt(modality, age, int(tumor_size))
    
    # other meta files
    edge_file = tumor_file.replace('-brain-', '-edge-')
    assert os.path.exists(edge_file)
    
    meta = {
        'patient_id': str(patient_id),
        'modality': str(modality),
        'slice': str(slice),
        'tumor_size': str(tumor_size),
        'age': str(age) if age is not None else "unknown",
        'image': str(tumor_file),
        'edge': str(edge_file),
        'prompt': str(prompt)
    }

    save_file = tumor_file.split('/')[-1].replace('.pkl', '.json').replace('-brain', '')
    with open(os.path.join(tumor_meta_dir, save_file), 'w') as f:
        json.dump(meta, f, indent=2)

# healthy
healthy_dir = os.path.join(config['data_root'], 'healthy')
healthy_files = [os.path.join(healthy_dir, f) for f in os.listdir(healthy_dir) if '-brain-' in f]
healthy_meta_dir = os.path.join(meta_dir, 'healthy')
os.makedirs(healthy_meta_dir, exist_ok=True)

for healthy_file in tqdm.tqdm(healthy_files):
    patient_id, _, modality, slice = healthy_file.split('/')[-1].split('-')
    slice = slice.replace('.pkl', '')

    try:
        age = int(survival_info[survival_info['patient_id'] == patient_id]['Age'].iloc[0])
    except (IndexError, ValueError):
        age = None

    prompt = generate_prompt(modality, age, 0)

    # other meta files
    edge_file = healthy_file.replace('-brain-', '-edge-')
    assert os.path.exists(edge_file)

    meta = {
        'patient_id': str(patient_id),
        'modality': str(modality),
        'slice': str(slice),
        'age': str(age) if age is not None else "unknown",
        'image': str(healthy_file),
        'edge': str(edge_file),
        'prompt': str(prompt)
    }

    save_file = healthy_file.split('/')[-1].replace('.pkl', '.json').replace('-brain', '')  
    with open(os.path.join(healthy_meta_dir, save_file), 'w') as f:
        json.dump(meta, f, indent=2)
