import os
import json
import numpy as np
import pickle
import tqdm
import torch

from collections import defaultdict

from torch.utils.data import Dataset
from monai.transforms import (
    Compose, EnsureChannelFirst, Resize, CastToType, RepeatChannel, Lambda
)
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedGroupKFold

from utils.util import set_env


class BraTSProcessor(object):
    def __init__(self,
                 config: dict = None,
                 tokenizer: AutoTokenizer = None):
        
        self.config = config
        self.tokenizer = tokenizer

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
        tumor_files = [f for f in os.listdir(self.tumor_dir) if 't1ce' in f]
        healthy_files = [f for f in os.listdir(self.healthy_dir) if 't1ce' in f]
        
        # Convert to full paths
        tumor_files = [os.path.join(self.tumor_dir, f) for f in tumor_files]
        healthy_files = [os.path.join(self.healthy_dir, f) for f in healthy_files]
        
        all_files = tumor_files + healthy_files
        labels = [1] * len(tumor_files) + [0] * len(healthy_files)
        # Extract patient IDs from full paths
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
        data_info = self.tokenize_prompts(data_info, self.tokenizer)
        
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
    
    def tokenize_prompts(self, data_info: dict, tokenizer: AutoTokenizer):
        """Tokenize prompts and add labels to data_info."""
        np.random.seed(self.seed)

        for split in ['train', 'test']:
            for condition in ['tumor', 'healthy']:
                label = 1 if condition == 'tumor' else 0
                data_batch = data_info[split][condition]
                prompts = []
                for data in data_batch:
                    p = np.random.random()
                    if split == 'train' and p < self.config['proportion_empty_prompts']:
                        prompts.append("")
                    else:
                        prompts.append(data['prompt'])
                # batch tokenize
                input_ids = tokenizer(
                    prompts,
                    return_tensors='pt',
                    padding='max_length',
                    truncation=True,
                    max_length=tokenizer.model_max_length
                ).input_ids  # shape: (batch, seq_len)

                for data, input_id, prompt in zip(data_batch, input_ids, prompts):
                    data['input_id'] = input_id
                    data['label'] = label
                    data['prompt'] = prompt

        return data_info


def create_transforms(resolution: int = 512, train: bool = True):
    assert resolution % 8 == 0, f"resolution must be divisible by 8, but got {resolution}"
        
    image_transform_list = [
        EnsureChannelFirst(channel_dim="no_channel"),
        CastToType(dtype=torch.float32),
        Resize((resolution, resolution), mode="bilinear"),
        Lambda(lambda x: x.clamp_(0.0, 1.0)),
        RepeatChannel(repeats=3),
    ]
    if train:
        image_transform_list.append(Lambda(lambda x: x * 2.0 - 1.0))
    
    conditioning_transform_list = [
        EnsureChannelFirst(channel_dim="no_channel"),
        CastToType(dtype=torch.float32),
        Resize((resolution, resolution), mode="nearest"),
        Lambda(lambda x: (x > 0.5).float()),
    ]

    image_transforms = Compose(image_transform_list)
    conditioning_transforms = Compose(conditioning_transform_list)

    return image_transforms, conditioning_transforms


def brats_collate_fn(batch):
    return {
        'idx': [item['idx'] for item in batch],
        'image': torch.stack([item['image'] for item in batch]),
        'edge': torch.stack([item['edge'] for item in batch]),
        'input_id': torch.stack([item['input_id'] for item in batch]),  # shape: (B, L)
        'label': torch.tensor([item['label'] for item in batch]),
        'prompt': [item['prompt'] for item in batch],
    }


class BraTSDataset(Dataset):
    def __init__(self,
                 data_info: dict,
                 image_transforms: Compose,
                 conditioning_transforms: Compose
        ):
        self.data_info = data_info
        self.image_transforms = image_transforms
        self.conditioning_transforms = conditioning_transforms

    def __getitem__(self, idx):
        
        data = self.data_info[idx]

        with open(data['image'], 'rb') as f:
            image = pickle.load(f)
        with open(data['edge'], 'rb') as f:
            edge = pickle.load(f)

        image = self.image_transforms(image)
        edge = self.conditioning_transforms(edge)
        input_id = data['input_id']
        prompt = data['prompt']
        label = data['label']
        return dict(image=image, edge=edge, input_id=input_id, label=label, prompt=prompt, idx=idx)
    
    def __len__(self):
        return len(self.data_info)


if __name__ == '__main__':
    
    from torch.utils.data import DataLoader
    from accelerate import Accelerator
    
    # 0. set environment
    config = {'server': 'psc',
              'proportion_empty_prompts': 0.5,
              'modality': 't1ce',
              'n_splits': 10,
              'fold_index': 0,
              'seed': 2025,
              'max_train_samples': 4000}
    config = set_env(config)

    # 1. prepare dataset
    tokenizer = AutoTokenizer.from_pretrained('runwayml/stable-diffusion-v1-5',
                                              subfolder='tokenizer',
                                              cache_dir=config['cache_dir'])
    processor = BraTSProcessor(config=config, tokenizer=tokenizer)

    accelerator = Accelerator()
    with accelerator.main_process_first():
        data_info = processor.process()

    train_data_info = data_info['train']['tumor'] + data_info['train']['healthy']
    import pickle
    with open(train_data_info[0]['image'], 'rb') as f:
        image = pickle.load(f)
    with open(train_data_info[0]['edge'], 'rb') as f:
        edge = pickle.load(f)
    print("image.shape", image.shape)
    print("edge.shape", edge.shape)

    # 2. create dataset
    image_transforms, conditioning_transforms = create_transforms(resolution=512, train=True)
    train_set = BraTSDataset(train_data_info, image_transforms, conditioning_transforms)

    # 3. create dataloader
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, collate_fn=brats_collate_fn)
    for batch in train_loader:
        print("batch['idx']", batch['idx'])
        print("batch['image'].shape", batch['image'].shape)
        print("batch['edge'].shape", batch['edge'].shape)
        print("batch['input_id']", batch['input_id'])
        print("batch['label']", batch['label'])
        print("batch['prompt']", batch['prompt'])
        break
    
    image_transforms, conditioning_transforms = create_transforms(resolution=512, train=False)
    test_set = BraTSDataset(data_info['test']['tumor'], image_transforms, conditioning_transforms)
    num_validation_samples = 2
    np.random.seed(config['seed'])
    test_idx = np.random.choice(test_set.__len__(), size=num_validation_samples, replace=False)
    print(test_idx)
    