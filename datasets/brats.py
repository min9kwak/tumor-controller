from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import os
import pickle
import numpy as np


class BraTSProcessor(object):
    def __init__(self,
                 data_root: str = 'D:/data/tumor-controller',
                 modality: str = 't1ce',
                 validation_size: float = 0.1,
                 test_size: float = 0.1,
                 random_state: int = 2023):

        assert modality in ['t1ce', 't2', 't1ce+t2']

        self.tumor_dir = os.path.join(data_root, 'normal')
        self.healthy_dir = os.path.join()

        self.modality: list = modality.split(modality)
        self.validation_size = validation_size
        self.test_size = test_size
        self.random_state = random_state

    def process(self):
        """
        Process the tumor and healthy directories to create datasets.
        Returns two datasets: tumor_set and healthy_set.
        """
        tumor_set = self.create_dataset(self.tumor_dir)
        healthy_set = self.create_dataset(self.healthy_dir)

        return dict(tumor=tumor_set, healthy=healthy_set)

    def create_dataset(self, data_dir):
        """
        Create a dataset from the given directory (tumor or healthy).
        Each entry includes image paths for the specified modalities and the corresponding segmentation path.
        """
        dataset = []

        for filename in os.listdir(data_dir):
            # Extract patient ID and modality from the filename
            patient_id, modality = filename.split('-')[0], filename.split('-')[-1].split('.')[0]

            # Process only if the modality matches
            if modality in self.modality:
                seg_filename = f"{patient_id}-{filename.split('-')[1]}-seg.pkl"
                seg_path = os.path.join(data_dir, seg_filename)

                # Ensure the segmentation file exists
                if os.path.exists(seg_path):
                    dataset.append({
                        "image": os.path.join(data_dir, filename),
                        "seg": seg_path,
                        "number": f"{patient_id}-{filename.replace('.pkl', '')}"
                    })

        return dataset


class BraTSDataset(Dataset):
    # TODO: remove test_flag
    def __init__(self, dataset, image_transform, seg_transform, label):

        self.dataset = dataset
        self.image_transform = image_transform
        self.seg_transform = seg_transform
        self.label = label

    def __getitem__(self, idx):

        with open(self.dataset[idx]['image'], 'rb') as fb:
            image = pickle.load(fb)
        with open(self.dataset[idx]['seg'], 'rb') as fb:
            seg = pickle.load(fb)

        if self.image_transform:
            image = self.image_transform(image)
        if self.seg_transform:
            seg = self.seg_transform(seg)

        out = dict(x=image, seg=seg, y=self.label, idx=idx)

        return out

    
if __name__ == '__main__':

    processor = BraTSProcessor()
    datasets = processor.process()

    batch_size = 4

    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    test_set = BraTSDataset(dataset=datasets['test'], transform=None, masking='random_patch',
                            patch_size=4, patch_ratio=0.3)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    for batch in test_loader:
        for idx in range(batch_size):
            plt.imshow(batch['x'][idx][0], cmap='gray')
            plt.imshow(batch['seg'][idx][0], cmap='Accent', alpha=0.2)
            plt.imshow(batch['mask'][idx][0], cmap='Reds', alpha=0.5)
            plt.show()
            import torch
            if torch.sum(batch['mask'][idx][0]).item() != 0:
                break
        break
