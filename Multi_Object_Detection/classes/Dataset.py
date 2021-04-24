import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.utils import make_grid
from torchvision.transforms import transforms
from torchvision.transforms import functional as TF

from Multi_Object_Detection.classes.utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class COCO_Dataset(Dataset):
    def __init__(self, BASE_DIR, transform=None, trans_params=None):
        super(COCO_Dataset, self).__init__()
        self.BASE_DIR = BASE_DIR
        path2listFile = os.path.join(BASE_DIR, "trainvalno5k.txt")
        with open(path2listFile, "r") as file:
            path2imgs = file.read().split('\n')[:-1]
        new_path2img = []
        for img in path2imgs:
            new_path = os.path.join(BASE_DIR, img[1:])
            new_path2img.append(new_path)

        self.path2imgs = new_path2img

        self.path2labels = [path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
                            for path in self.path2imgs]

        self.trans_params = trans_params
        self.transform = transform

    def __len__(self):
        return len(self.path2imgs)

    def __getitem__(self, item):
        path2img = self.path2imgs[item]
        # path2img = self.path2imgs[item % len(self.path2imgs)].rstrip()
        img = Image.open(path2img).convert('RGB')

        path2label = self.path2labels[item]
        # path2label = self.path2labels[item % len(self.path2imgs)].rstrip()
        labels = None

        if os.path.exists(path2label):
            labels = np.loadtxt(path2label).reshape(-1, 5)
        if self.transform:
            img, labels = self.transform(img, labels, self.trans_params)

        return img, labels, path2img

    def className(self, path):
        with open(os.path.join(path, 'coco.names'), "r") as file:
            content = file.read()
        class_names = content.split('\n')
        return class_names


def pad_to_square(img, boxes, pad_value=0, normalized_labels=True):
    w, h = img.size
    w_factor, h_factor = (w, h) if normalized_labels else (1, 1)

    dim_diff = np.abs(h - w)
    pad1 = dim_diff // 2
    pad2 = dim_diff - pad1

    if h <= w:
        left, top, right, bottom = 0, pad1, 0, pad2
    else:
        left, top, right, bottom = pad1, 0, pad2, 0

    padding = [left, top, right, bottom]

    img_padded = TF.pad(img=img, padding=padding, fill=pad_value)

    w_padded, h_padded = img_padded.size

    x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
    y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
    x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
    y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)

    x1 += padding[0]  # left
    y1 += padding[1]  # top
    x2 += padding[2]  # right
    y2 += padding[3]  # bottom

    boxes[:, 1] = ((x1 + x2) / 2) / w_padded
    boxes[:, 2] = ((y1 + y2) / 2) / h_padded
    boxes[:, 3] *= w_factor / w_padded
    boxes[:, 4] *= h_factor / h_padded

    return img_padded, boxes


def hflip(img, labels):
    img = TF.hflip(img)
    labels[:, 1] = 1.0 - labels[:, 1]
    return img, labels


def transform(img, labels, params):
    if params["pad2square"]:
        img, labels = pad_to_square(img, labels, pad_value=0,
                                    normalized_labels=params["normalized_labels"])

    img = TF.resize(img, params["target_size"])

    if np.random.rand() < params["p_hflip"]:
        img, labels = hflip(img, labels)

    img = TF.to_tensor(img)
    targets = torch.zeros((len(labels), 6))
    targets[:, 1:] = torch.from_numpy(labels)

    # targets = torch.from_numpy(labels)
    # targets = torch.unsqueeze(0)

    return img, targets


def dataLoader(path):
    trans_params_train = dict(target_size=(416, 416),
                              pad2square=True,
                              p_hflip=0.5,
                              normalized_labels=True)
    trans_params_val = dict(target_size=(416, 416),
                            pad2square=True,
                            p_hflip=0.0,
                            normalized_labels=True)

    coco_train = COCO_Dataset(path, transform=transform, trans_params=trans_params_train)
    coco_val = COCO_Dataset(path, transform=transform, trans_params=trans_params_val)

    train_dl = torch.utils.data.DataLoader(coco_train, batch_size=2, shuffle=True,
                                           num_workers=0, pin_memory=True,
                                           collate_fn=collat_fn)
    val_dl = torch.utils.data.DataLoader(coco_val, batch_size=4, shuffle=False,
                                         num_workers=0, pin_memory=True,
                                         collate_fn=collat_fn)
    # del coco_train
    # del coco_val
    # torch.cuda.empty_cache()
    return train_dl, val_dl, coco_train, coco_val

def collat_fn(batch):
    imgs, targets, paths = list(zip(*batch))
    # remove empty boxes
    targets = [boxes for boxes in targets if boxes is not None]
    # set the sample index
    for b_i, boxes in enumerate(targets):
        boxes[:, 0] = b_i
    targets = torch.cat(targets, 0)
    imgs = torch.stack([img for img in imgs])
    return imgs, targets, paths


if __name__ == '__main__':
    path = "../../Data/coco"
    train_dl, val_dl, coco_train, coco_val = dataLoader(path)

    for img, label, path_b in train_dl:
        break
    print(img.shape, label.shape)
    img, label, path_b = next(iter(val_dl))
    print(img.shape, label.shape)

