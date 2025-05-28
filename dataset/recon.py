import os
import numpy as np
import pickle

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from monai.transforms import ToTensor, ScaleIntensity, Compose, Transform
from collections import defaultdict


class ReconProcessor(object):

    def __init__(self,
                 data_root: str = 'D:/data/tumor-controller/',
                 healthy_data: str = None,
                 tumor_data: str = None,
                 modality: str = 't2',
                 validation_size: float = 0.1,
                 test_size: float = 0.1,
                 random_state: int = 2025):

        self.data_root = data_root
        self.healthy_data = healthy_data
        self.tumor_data = tumor_data

        self.modality = modality

        self.validation_size = validation_size
        self.test_size = test_size
        self.random_state = random_state

    def process(self):

        healthy_set = {'train': [], 'validation': [], 'test': []}
        tumor_set = {'train': [], 'validation': [], 'test': []}

        if self.healthy_data is not None:
            healthy_dir = os.path.join(self.data_root, 'healthy', self.healthy_data)
            assert os.path.isdir(healthy_dir), 'Healthy directory not found'
            healthy_filenames = [os.path.join(healthy_dir, f) for f in os.listdir(healthy_dir)]
            healthy_set = self.split_data(healthy_filenames, self.healthy_data)

        if self.tumor_data is not None:
            tumor_dir = os.path.join(self.data_root, 'tumor', self.tumor_data)
            assert os.path.isdir(tumor_dir), 'Tumor directory not found'
            tumor_filenames = [os.path.join(tumor_dir, f) for f in os.listdir(tumor_dir)]
            tumor_set = self.split_data(tumor_filenames, self.tumor_data)

        # Combine healthy & tumor
        dataset = {
            'train': healthy_set['train'] + tumor_set['train'],
            'validation': healthy_set['validation'] + tumor_set['validation'],
            'test': healthy_set['test'] + tumor_set['test']
        }
        return dataset

    def split_data(self, filenames, data_name):

        # unique patient ids
        patient_ids = list(set(self.extract_patient_id(f) for f in filenames))

        train_size = 1 - (self.validation_size + self.test_size)
        train_ids, test_ids = train_test_split(patient_ids, test_size=self.test_size, random_state=self.random_state)
        train_ids, validation_ids = train_test_split(
            train_ids, test_size=self.validation_size / (train_size + self.validation_size),
            random_state=self.random_state
        )

        # file matching: train, validation, test
        train_filenames = self.match_filenames(
            filenames=[f for f in filenames if self.extract_patient_id(f) in train_ids],
            data_name=data_name
        )
        validation_filenames = self.match_filenames(
            filenames=[f for f in filenames if self.extract_patient_id(f) in validation_ids],
            data_name=data_name
        )
        test_filenames = self.match_filenames(
            filenames=[f for f in filenames if self.extract_patient_id(f) in test_ids],
            data_name=data_name
        )

        return dict(train=train_filenames, validation=validation_filenames, test=test_filenames)

    def match_filenames(self, filenames, data_name):
        file_dict = defaultdict(lambda: {"img": None, "seg": "none"})

        if (data_name == 'ixi') and (self.modality == 't1ce+t2'):
            modality = 't1+t2'
        else:
            modality = self.modality

        for file in filenames:
            parts = file.split("\\")[-1].split("-")
            if len(parts) != 3 or not parts[2].endswith(".pkl"):
                continue

            patient_id, file_type, slice_num = parts
            slice_num = slice_num.replace(".pkl", "")

            key = f"{patient_id}-{slice_num}"

            if file_type == modality:
                file_dict[key]["img"] = file
            elif file_type == "seg":
                file_dict[key]["seg"] = file
            else:
                # TODO: filter image, seg files before
                continue

        return list(file_dict.values())

    @staticmethod
    def extract_patient_id(filename):
        return filename.split("\\")[-1].split("-")[0]


class ReconDataset(Dataset):

    def __init__(self, dataset, transform: Transform, return_seg: bool):
        self.dataset = dataset
        self.transform = transform
        self.return_seg = return_seg

    def __getitem__(self, index):

        image_filename = self.dataset[index]["img"]
        seg_filename = self.dataset[index]["seg"]

        # define label (healthy or tumor) by filename
        if 'healthy\\' in image_filename:
            weak_label = 0
        else:
            weak_label = 1

        # load image
        with open(image_filename, 'rb') as f:
            image = pickle.load(f)

        # load segmenation
        if self.return_seg:
            if seg_filename != 'none':
                with open(seg_filename, 'rb') as f:
                    seg = pickle.load(f)
            else:
                seg = np.zeros(shape=(1, 256, 256))

        if self.transform is not None:
            image = self.transform(image)

        out = dict(x=image, y=weak_label, out_dict={'y': weak_label}, idx=index, image_filename=image_filename)
        if self.return_seg:
            out['seg'] = seg
            out['seg_filename'] = seg_filename

        return out

    def __len__(self):
        return len(self.dataset)


def create_transforms():

    base_transform = [
        ToTensor(),
        ScaleIntensity(channel_wise=True)
    ]
    train_transform, test_transform = base_transform.copy(), base_transform.copy()

    # add some

    return Compose(train_transform), Compose(test_transform)


if __name__ == '__main__':

    from torch.utils.data import DataLoader

    processor = ReconProcessor(healthy_data='ixi', tumor_data='brats',
                               modality='t2')
    datasets = processor.process()

    train_dataset = ReconDataset(datasets['train'], create_transforms()[0], return_seg=True)
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)

    import tqdm
    import torch

    for batch in tqdm.tqdm(train_dataloader):
        # print(batch.keys())
        # print(batch['x'].shape)
        # print(batch['out_dict'])
        assert batch['x'][:, 0, :, :].min().item() == 0.0
        assert batch['x'][:, 0, :, :].max().item() == 1.0
        batch['image_filename']
        batch['seg_filename']
        batch['y']
        torch.sum(batch['seg'], dim=(1, 2, 3))
        break

    normal_processor = ReconProcessor(healthy_data='ixi')
    tumor_processor = ReconProcessor(tumor_data='brats')
    processors = [normal_processor, tumor_processor]
    for processor in processors:
        datasets = processor.process()
        train_dataset = ReconDataset(datasets['train'], create_transforms()[0], return_seg=False)
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        for batch in tqdm.tqdm(train_dataloader):
            import matplotlib.pyplot as plt
            plt.imshow(batch['x'].cpu().numpy()[0, 1, :, :], cmap='grey')
            plt.colorbar()
            plt.show()
            break
