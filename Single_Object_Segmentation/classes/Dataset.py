"""
link for dataset:
https://zenodo.org/record/1322001#.XcX1jk9KhhE
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from scipy import ndimage as ndi
from PIL import Image, ImageDraw

from skimage.segmentation import mark_boundaries
from sklearn.model_selection import ShuffleSplit
from albumentations import (HorizontalFlip, VerticalFlip,
                            Compose, Resize)
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, to_pil_image
import torchvision.transforms as transforms


class fetal_dataset(Dataset):
    def __init__(self, BASE_DIR, transform=None):
        super(fetal_dataset, self).__init__()
        imgList, anntList = Img_annotation_list(BASE_DIR)
        train_path = 'training_set'
        self.path2img = [os.path.join(BASE_DIR, train_path, fn)
                         for fn in imgList]
        self.path2annt = [os.path.join(BASE_DIR, train_path, fn)
                          for fn in anntList]
        self.transform = transform

    def __len__(self):
        return len(self.path2img)

    def __getitem__(self, item):
        path2img = self.path2img[item]
        img = Image.open(path2img)

        path2annt = self.path2annt[item]
        annt_edge = Image.open(path2annt)

        mask = ndi.binary_fill_holes(annt_edge)
        img = np.array(img)
        mask = mask.astype("uint8")

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        image = to_tensor(img)
        mask = to_tensor(mask) * 255
        return image, mask


def Img_annotation_list(path):
    list_files = os.listdir(os.path.join(path, 'training_set'))[:-1]

    imgList = [pp for pp in list_files if 'Annotation' not in pp]
    anntList = [pp for pp in list_files if 'Annotation' in pp]

    return imgList, anntList


def show_img_mask(img, mask):
    img_mask = mark_boundaries(np.array(img), np.array(mask), outline_color=(0, 1, 0), color=(0, 1, 0))
    plt.imshow(img_mask)

def data_train_val(BASE_DIR):
    h, w = 128, 192

    transform_train = Compose([
        Resize(h, w),
        HorizontalFlip(p=1),
        VerticalFlip(p=0.5)
    ])
    transform_val = Resize(h, w)

    fetal_ds1 = fetal_dataset(BASE_DIR, transform_train)
    fetal_ds2 = fetal_dataset(BASE_DIR, transform_val)

    sss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

    indices = range(len(fetal_ds1))

    for train_indx, val_indx in sss.split(indices):
        continue

    train_ds = torch.utils.data.Subset(fetal_ds1, train_indx)
    val_ds = torch.utils.data.Subset(fetal_ds2, val_indx)

    return train_ds, val_ds

def dataLoader(BASE_DIR):
    train_ds, val_ds = data_train_val(BASE_DIR)

    train_dl = torch.utils.data.DataLoader(dataset=train_ds,
                                           batch_size=8,
                                           shuffle=True,
                                           pin_memory=True)
    val_dl = torch.utils.data.DataLoader(dataset=val_ds,
                                         batch_size=16,
                                         shuffle=False,
                                         pin_memory=True)
    return train_dl, val_dl, train_ds, val_ds

if __name__ == '__main__':
    BASE_DIR = '../../Data/FETAL'
    train_path = 'training_set'
    imgList, anntList = Img_annotation_list(BASE_DIR)
    # print(len(imgList))
    # print(len(anntList))

    # rndImg = np.random.choice(imgList, 4)
    # plt.figure()
    # for i, fn in enumerate(rndImg):
    #     path2img = os.path.join(BASE_DIR, train_path, fn)
    #     path2annt = os.path.join(BASE_DIR, train_path, fn.replace('.png', '_Annotation.png'))
    #
    #     img = Image.open(path2img)
    #     annt_edge = Image.open(path2annt)
    #     mask = ndi.binary_fill_holes(annt_edge)
    #     i += 1
    #     print(i)
    #     plt.subplot(i, 3, 1)
    #     plt.imshow(img, cmap="gray")
    #
    #     plt.subplot(i, 3, 2)
    #     plt.imshow(mask, cmap='gray')
    #
    #     plt.subplot(i, 3, 3)
    #     show_img_mask(img, mask)
    # plt.show()


    train_dl, val_dl, train_ds, val_ds = dataLoader(BASE_DIR)

    img, mask = next(iter(train_dl))
    print(img.shape, mask.shape)
    show_img_mask(to_pil_image(img[0]), to_pil_image(mask[0]))
    plt.show()

    img, mask = next(iter(val_dl))
    print(img.shape, mask.shape)
    show_img_mask(to_pil_image(img[0]), to_pil_image(mask[0]))
    plt.show()