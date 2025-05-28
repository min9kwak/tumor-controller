import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
from monai import transforms as transforms
from monai.inferers import sliding_window_inference


class BraTSSegmentDataset(Dataset):

    def __init__(self, dataset, seg_classes=2, transform=None):
        self.dataset = dataset
        self.seg_classes = seg_classes
        self.transform = transform

    def __getitem__(self, idx):
        with open(self.dataset[idx]['image'], 'rb') as fb:
            image = pickle.load(fb)
        with open(self.dataset[idx]['label'], 'rb') as fb:
            label = pickle.load(fb)

        image = image / image.max()

        if self.transform:
            image = self.transform(image)

        label = self.one_hot_encode(label)

        return image, label

    def one_hot_encode(self, mask):
        """
        Convert labels to multi channels based on brats18 classes:
        label 1 is the necrotic and non-enhancing tumor core
        label 2 is the peritumoral edema
        label 4 is the GD-enhancing tumor
        The possible classes are TC (Tumor core), WT (Whole tumor)
        and ET (Enhancing tumor).
        """

        # if img has channel dim, squeeze it
        if mask.ndim == 4 and mask.shape[0] == 1:
            mask = mask.squeeze(0)

        if self.seg_classes == 4:
            # merge labels 1 (tumor non-enh) and 4 (tumor enh) and 2 (large edema) to WT
            # label 4 is ET
            result = [(mask == 1) | (mask == 4),
                      (mask == 1) | (mask == 4) | (mask == 2),
                      mask == 4]
        elif self.seg_classes == 2:
            result = [(mask == 1) | (mask == 2) | (mask == 4)]
        else:
            raise ValueError

        result = torch.stack(result, dim=0) if isinstance(mask, torch.Tensor) else np.stack(result, axis=0)

        return result.squeeze(1)

    def __len__(self):
        return len(self.dataset)


# evaluation
def make_inference_transforms(threshold: float = 0.5, use_sigmoid: bool = True, use_softmax: bool = False):
    return transforms.Compose([transforms.Activations(sigmoid=use_sigmoid, softmax=use_softmax),
                               transforms.AsDiscrete(threshold=threshold)])


if __name__ == '__main__':

    from torch.utils.data import DataLoader
    from datasets.brats import BraTSProcessor
    from model.segmentation import create_segmentation_model

    processor = BraTSProcessor(modality='all')
    datasets = processor.process()

    seg_classes = 3
    train_set = BraTSSegmentDataset(datasets['train'], seg_classes=seg_classes)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=False)

    for batch in train_loader:
        image, label = batch
        if label[0].sum() > 1000:
            import matplotlib.pyplot as plt

            plt.imshow(label.int()[0][0].numpy(), cmap='binary')
            plt.show()
            plt.imshow(label.int()[0][1].numpy(), cmap='Reds')
            plt.show()
            plt.imshow(label.int()[0][2].numpy(), cmap='Blues')
            plt.show()
            break

    seg_classes = 2
    train_set = BraTSSegmentDataset(datasets['train'], seg_classes=seg_classes)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=False)

    for batch in train_loader:
        image, label = batch
        if label[0].sum() > 1000:
            import matplotlib.pyplot as plt

            plt.imshow(label.int()[0][0].numpy(), cmap='binary')
            plt.show()
            break

    model = create_segmentation_model('unet', 4, 1)

    from monai.losses import DiceLoss
    from monai.metrics import DiceMetric
    from monai.data import decollate_batch

    dice_metric = DiceMetric(include_background=True, reduction="mean", num_classes=2)
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch", num_classes=2)

    criterion = DiceLoss(include_background=True, sigmoid=True, squared_pred=True)
    from monai.networks.nets import SwinUNETR
    def inference(input):
        out = sliding_window_inference(inputs=input, roi_size=(256, 256), sw_batch_size=1, predictor=model, overlap=0.5)
        return out

    post_trans = make_inference_transforms(threshold=0.5)

    for i, batch in enumerate(train_loader):
        image, label = batch
        image, label = image.float(), label.float()

        with torch.no_grad():
            pred = model(image)

        loss = criterion(pred, label)
        with torch.no_grad():
            outputs = inference(image)

        outputs = [post_trans(o) for o in decollate_batch(outputs)]

        # label = torch.zeros_like(label)
        # outputs = [torch.zeros_like(o) for o in outputs]

        dice_metric(outputs, label)
        metric = dice_metric.aggregate().item()

        dice_metric_batch(outputs, label)
        if i == 5:
            break

    dice_score = dice_metric.aggregate().item()
    # (Background) / TC / WT / ET
    dice_batch = dice_metric_batch.aggregate()
