import random
import torch
from monai.transforms import (
    Transform, ToTensor, ScaleIntensity, SpatialPad, CenterSpatialCrop, RandRotate90, RandFlip, Compose
)
from monai.utils.enums import TransformBackends


class AddChannel(Transform):
    backend = [TransformBackends.TORCH]

    def __init__(self):
        pass

    def __call__(self, img: torch.Tensor):
        return img.unsqueeze(dim=0)


class RandomSlice(Transform):
    backend = [TransformBackends.TORCH]

    def __init__(self, orientations, image_size: int = 128, slice_range: float = 0.15):
        assert all(orientation in [0, 1, 2] for orientation in orientations), "Orientations must be in [0, 1, 2]"

        center = image_size // 2
        interval = int(image_size * slice_range)
        self.orientations = orientations
        self.slice_range = (center - interval, center + interval + 1)

    def __call__(self, img: torch.Tensor):
        dim = random.choice(self.orientations)
        point = random.choice(range(*self.slice_range))

        slice_indices = [slice(None)] * 4
        slice_indices[dim + 1] = point
        out = img[slice_indices]
        return out


class CenterSlice(Transform):
    backend = [TransformBackends.TORCH]

    def __init__(self, orientations: int, image_size: int = 128):
        self.center = image_size // 2
        assert orientations in [0, 1, 2]
        self.orientations = orientations

    def __call__(self, img: torch.Tensor):
        slice_indices = [slice(None)] * 4
        slice_indices[self.orientations + 1] = self.center
        out = img[slice_indices]
        return out


def make_transforms(image_size: int = 128,
                    orientations: tuple = (0, 1, 2),
                    slice_range: float = 0.15,
                    rotate: bool = False,
                    flip: bool = False,
                    prob: float = 0.5):

    base_transform = [ToTensor(),
                      ScaleIntensity(),
                      AddChannel(),
                      SpatialPad(spatial_size=(145, 145, 145)),
                      CenterSpatialCrop(roi_size=(image_size, image_size, image_size)),
                      RandomSlice(orientations=orientations, image_size=image_size, slice_range=slice_range)]

    train_transform, test_transform = base_transform.copy(), base_transform.copy()

    if rotate:
        train_transform.append(RandRotate90(prob=prob))
    if flip:
        train_transform.append(RandFlip(prob=prob))

    return Compose(train_transform), Compose(test_transform)


def make_center_transforms(image_size: int = 128, orientations: int = 0):

    center_transform = [ToTensor(), ScaleIntensity(), AddChannel(), SpatialPad(spatial_size=(145, 145, 145)),
                        CenterSpatialCrop(roi_size=(image_size, image_size, image_size)),
                        CenterSlice(orientations=orientations, image_size=image_size)]
    return Compose(center_transform)
