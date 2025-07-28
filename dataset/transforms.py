import random
import torch
import monai.transforms as mt
from monai.transforms.utils_morphological_ops import dilate


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
    """
    Create image and edge conditioning transforms for training and inference.
    """
    
    assert resolution % 8 == 0, f"resolution must be divisible by 8, but got {resolution}"
        
    image_transform_list = [
        mt.EnsureChannelFirst(channel_dim="no_channel"),
        mt.CastToType(dtype=torch.float32),
        mt.Resize((resolution, resolution), mode="bilinear"),
        mt.Lambda(lambda x: x.clamp_(0.0, 1.0)),
        mt.RepeatChannel(repeats=3),
    ]
    if train:
        image_transform_list.append(mt.Lambda(lambda x: x * 2.0 - 1.0))
    
    conditioning_transform_list = [
        mt.EnsureChannelFirst(channel_dim="no_channel"),
        mt.CastToType(dtype=torch.float32),
        mt.Resize((resolution, resolution), mode="nearest"),
        mt.Lambda(lambda x: (x > 0.5).float()),
        mt.RepeatChannel(repeats=3),
    ]

    image_transforms = mt.Compose(image_transform_list)
    conditioning_transforms = mt.Compose(conditioning_transform_list)

    return image_transforms, conditioning_transforms


class Dilate(mt.Transform):
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def __call__(self, x):
        if x.ndim == 3:  # [C, H, W]
            x = x.unsqueeze(0)
            x = dilate(x, self.kernel_size)
            return x.squeeze(0)
        elif x.ndim == 4:
            return dilate(x, self.kernel_size)
        else:
            raise ValueError(f"Unexpected shape: {x.shape}")


def create_mask_transforms(resolution: int = 512, dilation_size: int = 5, sigma: float = 5.0):
    # dilation
    dilate_transform = mt.Compose(
        [
            mt.EnsureChannelFirst(channel_dim="no_channel"),
            mt.CastToType(dtype=torch.float32),
            mt.Resize((resolution, resolution), mode="nearest"),
            Dilate(dilation_size),
            mt.Lambda(lambda x: (x > 0.5).float())
        ]
    )

    # dilation + gaussian blur
    blur_transform = mt.Compose(
        [
            mt.EnsureChannelFirst(channel_dim="no_channel"),
            mt.CastToType(dtype=torch.float32),
            mt.Resize((resolution, resolution), mode="nearest"),
            Dilate(dilation_size),
            mt.GaussianSmooth(sigma=sigma),
            mt.Lambda(lambda x: torch.clamp(x, 0.0, 1.0))
        ]
    )

    return dilate_transform, blur_transform
