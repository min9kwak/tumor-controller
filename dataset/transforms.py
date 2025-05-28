import random
import torch
from monai.transforms import Compose, EnsureChannelFirst, ToTensor, Resize
from monai.utils.enums import TransformBackends


def make_transforms(image_size: int = 512):
    
    image_transform = Compose([
        EnsureChannelFirst(),
        ToTensor(torch.float32),
        Resize(spatial_size=(image_size, image_size), mode='bilinear')
    ])
    seg_transform = Compose([
        EnsureChannelFirst(),
        ToTensor(torch.long),
        Resize(spatial_size=(image_size, image_size), mode='bilinear')
    ])

    return image_transform, seg_transform
