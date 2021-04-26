import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
from albumentations import (
    HorizontalFlip,
    Compose,
    Resize,
    Normalize
)

from Multi_Object_Segmentation.classes.Utils import *


class Dataset(VOCSegmentation):
    def __getitem__(self, item):
        img = Image.open(self.images[item]).convert('RGB')
        target = Image.open(self.masks[item])

        if self.transforms:
            augmented = self.transforms(image=np.array(img), mask=np.array(target))
            img = augmented['image']
            target = augmented['mask']
            target[target > 20] = 0

        img = to_tensor(img)
        target = torch.from_numpy(target).type(torch.long)
        return img, target


def dataset_ds(h, w, mean, std, path):
    transform_train = Compose([Resize(h, w),
                               HorizontalFlip(p=0.5),
                               Normalize(mean=mean, std=std)])
    transform_val = Compose([Resize(h, w),
                             Normalize(mean=mean, std=std)])

    train_ds = Dataset(path, year='2012', image_set='train', download=False,
                       transforms=transform_train)
    val_ds = Dataset(path, year='2012', image_set='val', download=False,
                     transforms=transform_val)
    return train_ds, val_ds


def dataset_dl(train_ds, val_ds):
    train_dl = DataLoader(train_ds, batch_size=2, shuffle=True, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=False, pin_memory=True)
    return train_dl, val_dl


if __name__ == '__main__':
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    h, w = 520, 520
    train_ds, val_ds = dataset_ds(h, w, mean, std)
    train_dl, val_dl = dataset_dl(train_ds, val_ds)
    print(len(train_dl.dataset), len(val_dl.dataset))

    imgs, masks = next(iter(train_dl))
    print(imgs.shape, masks.shape)
    img = imgs[0]
    mask = masks[0]
    print(img.shape, img.type(), torch.max(img))
    print(mask.shape, mask.type(), torch.max(mask))

    show_img_mask_boundaries(img, mask, mean, std)
