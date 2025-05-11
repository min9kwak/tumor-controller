from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import os
import pickle
import numpy as np


class BraTSProcessor(object):
    def __init__(self,
                 data_root: str = 'D:/data/BraTS/slice',
                 modality: str = 't1',
                 validation_size: float = 0.1,
                 test_size: float = 0.1,
                 random_state: int = 2023):

        assert modality in ['t1', 't1ce', 't2', 'flair', 'all', 't1ce+t2']
        self.data_root = data_root
        self.modality = modality
        self.validation_size = validation_size
        self.test_size = test_size
        self.random_state = random_state

    def process(self):

        patient_ids = os.listdir(self.data_root)
        train_size = 1 - (self.validation_size + self.test_size)

        train_ids, test_ids = train_test_split(patient_ids, test_size=self.test_size, random_state=self.random_state)
        train_ids, validation_ids = train_test_split(
            train_ids, test_size=self.validation_size / (train_size + self.validation_size),
            random_state=self.random_state
        )

        train_set = self.create_dataset(train_ids)
        validation_set = self.create_dataset(validation_ids)
        test_set = self.create_dataset(test_ids)

        return dict(train=train_set, validation=validation_set, test=test_set)

    def create_dataset(self, ids):

        dataset = []

        for id in ids:

            filenames = os.listdir(os.path.join(self.data_root, id))
            image_filenames = sorted([f for f in filenames if f'{self.modality}_' in f])
            seg_filenames = sorted([f for f in filenames if f'seg_' in f])

            # check slice sequences and pairing
            for image_filename, seg_filename in zip(image_filenames, seg_filenames):
                assert image_filename.split('_')[1] == seg_filename.split('_')[1]
                out = dict(image=os.path.join(self.data_root, id, image_filename),
                           label=os.path.join(self.data_root, id, seg_filename),
                           number=f"{id}-{image_filename.replace('.pkl', '')}")
                dataset.append(out)

        return dataset


class BraTSDataset(Dataset):
    # TODO: remove test_flag
    def __init__(self, dataset, transform, masking: str = None,
                 patch_size: int = 4, patch_ratio: float = 0.3):

        self.dataset = dataset
        self.transform = transform

        self.masking = masking
        self.patch_size = patch_size if masking is not None else None
        self.patch_ratio = patch_ratio if masking is not None else None

    def __getitem__(self, idx):

        with open(self.dataset[idx]['image'], 'rb') as fb:
            image = pickle.load(fb)
        with open(self.dataset[idx]['label'], 'rb') as fb:
            label = pickle.load(fb)

        # TODO: transform - at least rotate90

        weak_label = 1 if label.max() > 0 else 0

        out_dict = {}
        out_dict['y'] = weak_label

        number = self.dataset[idx]['number']

        image = image / image.max()

        # return data sample
        out = dict(x=image, y=weak_label, out_dict=out_dict, seg=label, idx=number)
        if self.masking is None:
            return out
        elif self.masking == 'half':
            mask = self.mask_half(label)
            out['mask'] = mask
        elif self.masking == 'random_patch':
            mask = self.mask_random_patch(image)
            out['mask'] = mask
        else:
            raise ValueError('masking must be either "half" or "random_patch"')
        return out

    def mask_random_patch(self, image):
        #
        C, H, W = image.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        num_patches = (H // self.patch_size) * (W // self.patch_size)
        num_zero_patches = int(num_patches * self.patch_ratio)

        mask = np.ones_like(image, dtype=image.dtype)
        indices = np.random.permutation(num_patches)[:num_zero_patches]
        for idx in indices:
            row = idx // (W // self.patch_size)
            col = idx % (W // self.patch_size)
            mask[:, row * self.patch_size:(row + 1) * self.patch_size,
                 col * self.patch_size:(col + 1) * self.patch_size] = 0.0

        return mask

    @staticmethod
    def mask_half(label: np.ndarray):
        H = label.shape[2]
        mid_line = H // 2

        tumor_count = np.array(label > 0, dtype=float)
        upper = tumor_count[..., :mid_line, :].sum()
        lower = tumor_count[..., mid_line:, :].sum()

        # tumorous area. 0: upper, 1: lower
        if upper == 0 and lower == 0:
            flag = np.random.randint(0, 2)
        elif upper >= lower:
            flag = 0
        else:
            flag = 1

        # create half mask (tumorous). 1 indicates tumorous area
        mask = np.zeros_like(label)
        if flag == 0:
            mask[..., :mid_line, :] = 1.0
        else:
            mask[..., mid_line:, :] = 1.0

        return mask

    def __len__(self):
        return len(self.dataset)


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
