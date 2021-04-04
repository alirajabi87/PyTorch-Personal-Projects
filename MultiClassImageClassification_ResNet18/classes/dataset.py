"""
We will use STL-10 dataset provided in the PyTorch torchvision package.
"""

import torch
from torch.utils.data import Subset
import torchvision.transforms as transforms
from torchvision import datasets
import torchvision
import os
import numpy as np
import matplotlib.pyplot as plt
import collections
from sklearn.model_selection import StratifiedShuffleSplit


def Counter(dataset):
    y0 = [y for _, y in dataset]
    _counter = collections.Counter(y0)
    return _counter


def show(img, y=None, color=True):
    img = np.transpose(img.numpy(), (1, 2, 0))
    plt.imshow(img)
    if y:
        plt.title(f"label: {str(y)}")
    plt.show()


def randomShow(dataset, grid_size=4):
    rand_inds = np.random.randint(0, len(dataset), grid_size)
    x_grid = [dataset[i][0] for i in rand_inds]
    y_grid = [dataset[i][1] for i in rand_inds]

    x_grid = torchvision.utils.make_grid(x_grid, nrow=grid_size, padding=1)
    plt.rcParams['figure.figsize'] = (grid_size * 2 + 2, grid_size * 2 + 2)
    show(x_grid, y_grid)


def transform_Type(dataset, train=True):
    meanRGB = [np.mean(x.numpy(), axis=(1, 2)) for x, _ in dataset]
    stdRGB = [np.std(x.numpy(), axis=(1, 2)) for x, _ in dataset]

    meanR = np.mean([m[0] for m in meanRGB])
    meanG = np.mean([m[1] for m in meanRGB])
    meanB = np.mean([m[2] for m in meanRGB])

    stdR = np.mean([s[0] for s in stdRGB])
    stdG = np.mean([s[1] for s in stdRGB])
    stdB = np.mean([s[2] for s in stdRGB])

    # print(meanR, meanG, meanB)
    # print(stdR, stdG, stdB)

    if train:
        transformer = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[meanR, meanG, meanB], std=[stdR, stdG, stdB])
        ])
    else:
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[meanR, meanG, meanB], std=[stdR, stdG, stdB])
        ])

    return transformer


def datasetLoader(dataset, batch_size=32, train=True):
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=True
    )


def dataGenerator():
    classes = dict(Airplane=0, Bird=1, Car=2, Cat=3,
                   Deer=4, Dog=5, Horse=6, Monkey=7, Ship=8, Truck=9)
    image_size = (96, 96)
    path = "../Data/STL10/"

    if not os.path.exists(path):
        os.mkdir(path)

    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.STL10(path, transform=transform, download=True, split='train')
    test_ds0 = datasets.STL10(path, transform=transform, download=True, split='test')

    # print(train_ds.data.shape)
    # print(test_ds.data.shape)
    print(Counter(train_ds))

    y_test0 = [y for _, y in test_ds0]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

    indices = list(range(len(test_ds0)))

    # test_ind, val_ind = sss.split(indices, y_test0)
    # print(len(val_ind), len(test_ind))
    for test_ind, val_ind in sss.split(indices, y_test0):
        # print(f"test: {test_ind}, val: {val_ind}")
        # print(len(val_ind), len(test_ind))
        continue

    val_ds = Subset(test_ds0, val_ind)
    test_ds = Subset(test_ds0, test_ind)

    train_ds.transform = transform_Type(train_ds, train=True)
    test_ds0.transform = transform_Type(test_ds0, train=False)

    # Checking
    # print(Counter(test_ds))
    # print(Counter(val_ds))
    # randomShow(train_ds, 4)
    # randomShow(val_ds, 4)
    # randomShow(test_ds, 4)

    train_dl = datasetLoader(train_ds, 32, True)
    test_dl = datasetLoader(test_ds, 64, False)
    val_dl = datasetLoader(val_ds, 64, False)

    # x, y = next(iter(train_dl))
    # print(x.shape, y.shape)

    return train_dl, val_dl, test_dl

if __name__ == '__main__':
    dataGenerator()
