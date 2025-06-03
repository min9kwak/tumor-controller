import os
import pickle
import torch

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class BraTSProcessor(object):
    def __init__(self,
                 data_root: str = 'D:/data/tumor-controller',
                 modality: str = 't1ce',
                 validation_size: float = 0.1,
                 test_size: float = 0.1,
                 random_state: int = 2023):

        assert modality in ['t1ce', 't2', 't1ce+t2']

        self.tumor_dir = os.path.join(data_root, 'tumor')
        self.healthy_dir = os.path.join(data_root, 'healthy')

        self.modality: list = modality.split(modality)
        self.validation_size = validation_size
        self.test_size = test_size
        self.random_state = random_state

    def process(self):
        """
        Process the tumor and healthy directories to create datasets.
        Returns two datasets: tumor_set and healthy_set.
        """
        tumor_set = self.create_dataset(self.tumor_dir, True)
        healthy_set = self.create_dataset(self.healthy_dir, False)

        return dict(tumor=tumor_set, healthy=healthy_set)

    def create_dataset(self, data_dir, is_tumor=True):
        """
        Create a dataset from the given directory (tumor or healthy).
        Each entry includes image paths for the specified modalities and the corresponding segmentation path.
        """
        dataset = []

        for filename in os.listdir(data_dir):
            # Extract patient ID and modality from the filename
            patient_id, modality, z = filename.replace('.pkl', '').split('-')

            # Process only if the modality matches
            if modality in self.modality:
                if is_tumor:
                    seg_filename = filename.replace('-seg-', f'{modality}')
                    seg_path = os.path.join(data_dir, seg_filename)
                else:
                    seg_path = 'healthy'

                # TODO: add edge map
                dataset.append(dict(image=os.path.join(data_dir, filename), seg=seg_path))

        return dataset



class BrainTumorControlNetDataset(Dataset):
    def __init__(self, json_path, image_size=512):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
            
        self.image_transforms = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        self.conditioning_transforms = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load and transform the target image (tumor)
        image = Image.open(item['image']).convert('RGB')
        pixel_values = self.image_transforms(image)
        
        # Load and transform the conditioning image (healthy)
        conditioning_image = Image.open(item['conditioning_image']).convert('RGB')
        conditioning_pixel_values = self.conditioning_transforms(conditioning_image)
        
        return {
            'pixel_values': pixel_values,
            'conditioning_pixel_values': conditioning_pixel_values,
            'text': item['text']
        }
    

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
        if self.image_transform:
            image = self.image_transform(image)
        
        # TODO: add edge map
        
        if self.label == 1:
            with open(self.dataset[idx]['seg'], 'rb') as fb:
                seg = pickle.load(fb)
            if self.seg_transform:
                seg = self.seg_transform(seg)
        elif self.label == 0:
            seg = torch.zeros_like(image, dtype=torch.long)

        out = dict(x=image, seg=seg, y=self.label, idx=idx)

        return out

    
if __name__ == '__main__':

    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from dataset.transforms import make_transforms
    
    processor = BraTSProcessor()
    datasets = processor.process()

    image_transform, seg_transform = make_transforms(image_size=512)

    batch_size = 2

    
    test_set = BraTSDataset(dataset=datasets['tumor']['test'],
                            image_transform=image_transform,
                            seg_transform=seg_transform,
                            label=1)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    for batch in test_loader:
        for idx in range(batch_size):
            plt.imshow(batch['x'][idx][0], cmap='gray')
            plt.imshow(batch['seg'][idx][0], cmap='Accent', alpha=0.2)
            plt.show()
        break
