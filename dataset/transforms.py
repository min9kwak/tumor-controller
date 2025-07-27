import random
import torch
from monai.transforms import (
    Compose, EnsureChannelFirst, Resize, CastToType, RepeatChannel, Lambda
)


def brats_collate_fn(batch):
    collated = {}
    keys = batch[0].keys()

    for key in keys:
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([item[key] for item in batch])
        elif isinstance(batch[0][key], (int, float)):
            collated[key] = torch.tensor([item[key] for item in batch])
        else:
            collated[key] = [item[key] for item in batch]

    return collated


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
        RepeatChannel(repeats=3),
    ]

    image_transforms = Compose(image_transform_list)
    conditioning_transforms = Compose(conditioning_transform_list)

    return image_transforms, conditioning_transforms
