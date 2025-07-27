import os
import json
import numpy as np
import pickle
import tqdm
import random

from collections import defaultdict

from torch.utils.data import Dataset
import monai.transforms as mt

from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedGroupKFold

from utils.util import set_env
from dataset.preprocessing.helper import PromptBuilder


class BraTSProcessor(object):
    def __init__(self, config: dict = None):
        
        self.config = config

        assert self.config['modality'] in ['t1ce', 't2', 't1ce+t2'], \
            "modality must be one of ['t1ce', 't2', 't1ce+t2']"
        assert self.config['proportion_empty_prompts'] >= 0 and self.config['proportion_empty_prompts'] <= 1, \
            "proportion_empty_prompts must be between 0 and 1"
        assert self.config['n_splits'] > 1, \
            "n_splits must be greater than 1"
        assert self.config['fold_index'] >= 0 and self.config['fold_index'] < self.config['n_splits'], \
            f"fold_index must be between 0 and {self.config['n_splits'] - 1}"
        
        modality = self.config['modality']
        self.modality = modality.split('+') if '+' in modality else [modality]

        self.tumor_dir = os.path.join(config['data_root'], 'meta', 'tumor')
        self.healthy_dir = os.path.join(config['data_root'], 'meta', 'healthy')        
        self.seed = self.config['seed']
        self.n_splits = self.config['n_splits']
        self.fold_index = self.config['fold_index']
    
    def load_data(self):
        """Load data files and perform basic labeling."""
        
        tumor_files = [f for f in os.listdir(self.tumor_dir) if any(m in f for m in self.modality)]
        healthy_files = [f for f in os.listdir(self.healthy_dir) if any(m in f for m in self.modality)]
        
        tumor_files = [os.path.join(self.tumor_dir, f) for f in tumor_files]
        healthy_files = [os.path.join(self.healthy_dir, f) for f in healthy_files]

        all_files = tumor_files + healthy_files
        labels = [1] * len(tumor_files) + [0] * len(healthy_files)
        groups = [os.path.basename(f).split('-')[0] for f in all_files]

        return all_files, labels, groups
    
    def split_data(self, all_files, labels, groups):
        """Split data into train/test sets and validate the split.
        
        Args:
            all_files: List of all file paths (full directory paths)
            labels: List of corresponding labels (1 for tumor, 0 for healthy)
            groups: List of patient IDs for stratification
            
        Returns:
            Dictionary containing train and test splits for both tumor and healthy cases
        """
        kfold = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        
        # Get the specified fold
        for i, (train_idx, test_idx) in enumerate(kfold.split(all_files, labels, groups)):
            if i == self.fold_index:
                break
        else:
            raise ValueError(f"Fold index {self.fold_index} is out of range")
            
        train_files = [all_files[i] for i in train_idx]
        test_files = [all_files[i] for i in test_idx]
        train_labels = [labels[i] for i in train_idx]
        test_labels = [labels[i] for i in test_idx]

        # max_train_samples
        if self.config.get('max_train_samples', None) is not None:
            np.random.seed(self.seed)
            selected_idx = np.random.choice(len(train_files),
                                            size=min(self.config["max_train_samples"], len(train_files)),
                                            replace=False)
            train_files = [train_files[i] for i in selected_idx]
            train_labels = [train_labels[i] for i in selected_idx]
        
        # Split by condition
        tumor_train = [f for f, y in zip(train_files, train_labels) if y == 1]
        healthy_train = [f for f, y in zip(train_files, train_labels) if y == 0]
        tumor_test = [f for f, y in zip(test_files, test_labels) if y == 1]
        healthy_test = [f for f, y in zip(test_files, test_labels) if y == 0]
        
        # Verify split validity using basenames
        train_pids = set([os.path.basename(f).split('-')[0] for f in train_files])
        test_pids = set([os.path.basename(f).split('-')[0] for f in test_files])
        overlap = train_pids & test_pids
        assert len(overlap) == 0
        
        if self.config.get('max_train_samples', None) is not None:
            assert self.config['max_train_samples'] == len(tumor_train) + len(healthy_train)
        else:
            assert len(all_files) == len(tumor_train) + len(healthy_train) + len(tumor_test) + len(healthy_test)
        
        data_filenames = {
            'train': {'tumor': tumor_train, 'healthy': healthy_train},
            'test': {'tumor': tumor_test, 'healthy': healthy_test}
        }
        data_info = self.load_data_info(data_filenames)
        
        return data_info
    
    def load_data_info(self, data_filenames: dict):
        """Load data info from file paths."""
        data_info = defaultdict(lambda: defaultdict(list))
        for split in ['train', 'test']:
            for condition in ['tumor', 'healthy']:
                file_names = data_filenames[split][condition]
                for file_name in tqdm.tqdm(file_names, desc=f'Loading {split}-{condition}'):
                    with open(file_name, 'r') as f:
                        data = json.load(f)
                    data_info[split][condition].append(data)
        return dict(data_info)

    def print_summary(self, data_info):
        for split in ['train', 'test']:
            t, h = len(data_info[split]['tumor']), len(data_info[split]['healthy'])
            print(f"{split.title()} - Tumor: {t}, Healthy: {h}, Total: {t + h}")
    
    def process(self):
        """Execute the complete data processing pipeline."""
        all_files, labels, groups = self.load_data()
        data_info = self.split_data(all_files, labels, groups)
        self.print_summary(data_info)
        return data_info


class BraTSDataset(Dataset):
    def __init__(self,
                 data_info: list,
                 prompt_builder: PromptBuilder,
                 image_transform=None,
                 conditioning_transform=None,
                 dilate_transform=None,
                 blur_transform=None,
                 proportion_empty_prompts=0.0,
                 return_keys=None):
        
        self.prompt_builder = prompt_builder
        self.image_transform = image_transform
        self.conditioning_transform = conditioning_transform
        self.dilate_transform = dilate_transform
        self.blur_transform = blur_transform
        self.proportion_empty_prompts = proportion_empty_prompts
        self.return_keys = return_keys or ['image', 'edge', 'dilate', 'blur', 'prompt', 'label', 'idx']

        self.tumor_seg_by_slice = self._index_tumor_segs_by_slice(data_info)
        self.data_info = self._filter_invalid_healthy_slices(data_info)

        # Define loader functions per field
        self.loaders = {
            'image': lambda d: self.image_transform(pickle.load(open(d['image'], 'rb'))),
            'edge': lambda d: self.conditioning_transform(pickle.load(open(d['edge'], 'rb'))),
            'dilate': lambda d: self.dilate_transform(self._load_seg(d)),
            'blur': lambda d: self.blur_transform(self._load_seg(d)),
            'prompt': lambda d: self._build_prompt(d),
            'label': lambda d: int(d['label']),  # <- int cast here
            'idx': lambda d: d['idx'],           # will be injected in __getitem__
        }

        invalid_keys = set(self.return_keys) - set(self.loaders.keys())
        assert not invalid_keys, f"Invalid return_keys: {invalid_keys}"

        self.loaders = {k: self.loaders[k] for k in self.return_keys}

    def _index_tumor_segs_by_slice(self, data_info):
        index = {}
        for d in data_info:
            if int(d['label']) == 1:
                slice_key = int(d['slice'])  # ← 변경
                index.setdefault(slice_key, []).append(d['seg'])
        return index

    def _filter_invalid_healthy_slices(self, data_info):
        filtered = []
        removed = 0
        for d in data_info:
            if int(d['label']) == 0 and d['slice'] not in self.tumor_seg_by_slice:
                removed += 1
                continue
            filtered.append(d)
        if removed > 0:
            print(f"[BraTSDataset] Removed {removed} invalid healthy samples.")
        return filtered

    def _load_seg(self, d):
        if int(d['label']) == 1:
            return pickle.load(open(d['seg'], 'rb'))
        else:
            slice_key = int(d['slice'])  # ← 변경
            if slice_key not in self.tumor_seg_by_slice:
                raise ValueError(f"No tumor seg found for healthy slice {slice_key}")
            candidates = self.tumor_seg_by_slice[slice_key]
            seg_path = random.choice(candidates)
            return pickle.load(open(seg_path, 'rb'))

    def _build_prompt(self, d):
        age = None if d['age'] == "unknown" else int(d['age'])
        tumor_size = int(d.get('tumor_size', 0))  # string → int, default=0 for healthy
        if random.random() < self.proportion_empty_prompts:
            return ""
        return self.prompt_builder.generate_prompt(d['modality'], age, tumor_size)

    def __getitem__(self, idx):
        d = self.data_info[idx]
        d['idx'] = idx  # inject index
        return {k: self.loaders[k](d) for k in self.return_keys}

    def __len__(self):
        return len(self.data_info)


if __name__ == '__main__':
    
    from torch.utils.data import DataLoader
    from accelerate import Accelerator
    from dataset.transforms import create_transforms, brats_collate_fn, create_mask_transforms
    from functools import partial

    # 0. set environment
    config = {'server': 'psc',
                'proportion_empty_prompts': 0.5,
                'modality': 't1ce+t2',
                'n_splits': 10,
                'fold_index': 0,
                'seed': 2025,
                'max_train_samples': 1000}
    config = set_env(config)

    # 1. prepare dataset
    tokenizer = AutoTokenizer.from_pretrained('runwayml/stable-diffusion-v1-5',
                                            subfolder='tokenizer',
                                            cache_dir=config['cache_dir'],
                                            use_fast=True)
    processor = BraTSProcessor(config=config)
    accelerator = Accelerator()

    with accelerator.main_process_first():
        data_info = processor.process()

    train_data_info = data_info['train']['tumor'] + data_info['train']['healthy']
    for k, v in train_data_info[0].items():
        print(f"{k}: {v}")

    with open(train_data_info[0]['image'], 'rb') as f:
        image = pickle.load(f)
    with open(train_data_info[0]['edge'], 'rb') as f:
        edge = pickle.load(f)
    with open(train_data_info[0]['seg'], 'rb') as f:
        seg = pickle.load(f)
    print("image.shape", image.shape)
    print("edge.shape", edge.shape)
    print("seg.shape", seg.shape)

    # 2. create dataset
    from dataset.preprocessing.helper import PromptBuilder
    from dataset.brats import BraTSDataset

    prompter = PromptBuilder()
    image_transforms, conditioning_transforms = create_transforms(resolution=512, train=True)
    dilate_transform, blur_transform = create_mask_transforms(resolution=512, dilation_size=5, sigma=5.0)

    train_set = BraTSDataset(train_data_info,
                            prompt_builder=prompter,
                            image_transform=image_transforms,
                            conditioning_transform=conditioning_transforms,
                            dilate_transform=dilate_transform,
                            blur_transform=blur_transform)

    # 3. create dataloader
    from dataset.transforms import brats_collate_fn
    collate_fn = partial(brats_collate_fn, tokenizer=tokenizer)

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, collate_fn=collate_fn)

    for batch in train_loader:
        print("batch['idx']:", batch['idx'])
        print("batch['image'].shape:", batch['image'].shape)
        print("batch['edge'].shape:", batch['edge'].shape)
        print("batch['dilate'].shape:", batch['dilate'].shape)
        print("batch['blur'].shape:", batch['blur'].shape)
        print("batch['input_ids'].shape:", batch['input_ids'].shape)
        print("batch['label']:", batch['label'])
        break

    # 4. test set for quick validation
    image_transforms, conditioning_transforms = create_transforms(resolution=512, train=False)
    dilate_transform, blur_transform = create_mask_transforms(resolution=512, dilation_size=5, sigma=5.0)

    test_set = BraTSDataset(data_info['test']['tumor'],
                            prompt_builder=prompter,
                            image_transform=image_transforms,
                            conditioning_transform=conditioning_transforms,
                            dilate_transform=dilate_transform,
                            blur_transform=blur_transform)

    np.random.seed(config['seed'])
    test_idx = np.random.choice(len(test_set), size=2, replace=False)
    print("Validation indices:", test_idx)

    # 5. check dilate/blur output for a sample
    print("\n[Checking dilate and blur transformations]")
    sample = train_set[0]

    import matplotlib.pyplot as plt

    if 'dilate' in sample and sample['dilate'] is not None:
        print("sample['dilate'].shape:", sample['dilate'].shape)
        print("sample['dilate'].min():", sample['dilate'].min(), "max():", sample['dilate'].max())
        plt.imshow(sample['dilate'].squeeze().cpu(), cmap='gray')
        plt.title('Dilated Mask')
        plt.axis('off')
        plt.show()

    if 'blur' in sample and sample['blur'] is not None:
        print("sample['blur'].shape:", sample['blur'].shape)
        print("sample['blur'].min():", sample['blur'].min(), "max():", sample['blur'].max())
        plt.imshow(sample['blur'].squeeze().cpu(), cmap='gray')
        plt.title('Blurred Mask')
        plt.axis('off')
        plt.show()