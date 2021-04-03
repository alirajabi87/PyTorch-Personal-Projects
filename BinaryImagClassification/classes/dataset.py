"""
https://www.kaggle.com/c/histopathologic-cancer-detection/data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, random_split, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import os
from PIL import Image, ImageDraw


class CheckandDraw:
    def __init__(self, path):
        self.path = path

    def check_df(self, df):
        print(df.head())
        print(df['label'].value_counts())
        df['label'].hist()
        plt.show()

    def drawPictures(self, df, nrows, ncols, info=True, data_type="train"):
        malignantIds = df.loc[df['label'] == 1]['id'].values
        plt.rcParams['figure.figsize'] = (nrows * ncols + 1.0, nrows * ncols + 1.0)
        plt.subplots_adjust(wspace=0, hspace=0)
        nrows, ncols = nrows, ncols

        for i, id in enumerate(malignantIds[:nrows * ncols]):
            full_filename = os.path.join(self.path, data_type + "/", id + ".tif")
            # if os.path.isfile(full_filename):
            #     print("found the file")
            # else:
            #     print("something is wrong")
            img = Image.open(full_filename)

            if info:
                img1 = np.array(img)
                print(f"Pixel values range from {np.min(img1)} to {np.max(img1)}")
                print(f"The image shape is: {img1.shape}")
                info = False

            draw = ImageDraw.Draw(img)
            draw.rectangle(((32, 32), (64, 64)), outline="green")
            plt.subplot(nrows, ncols, i + 1)
            plt.imshow(np.array(img))
            plt.axis('off')
        plt.show()


class HistoCancerDataset(Dataset):
    def __init__(self, data_dir, transform, data_type="train"):
        super(HistoCancerDataset, self).__init__()
        path2data = os.path.join(data_dir, data_type)

        filenames = os.listdir(path2data)

        self.full_names = [os.path.join(path2data, file) for file in filenames]
        self.transform = transform

        df = pd.read_csv(os.path.join(data_dir, data_type+"_labels.csv"))
        df.set_index('id', inplace=True)

        self.labels = [df.loc[filename[:-4]].values[0] for
                       filename in filenames]  # deleting .tif from the end of the file name

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        img = Image.open(self.full_names[item])
        img = self.transform(img)
        return img, self.labels[item]

    def train_test_split(self, ds, test_size=0.2):
        torch.manual_seed(0)
        train = int(len(self.labels) * (1 - test_size))
        test = int(len(self.labels) * test_size)
        train_ds, test_ds = random_split(ds, [train, test], generator=torch.Generator().manual_seed(1))
        return train_ds, test_ds

    def showImage(self, img, y):
        img = np.transpose(img.numpy(), (1, 2, 0))
        plt.imshow(img, interpolation='nearest')
        plt.title("label:" + str(y))
        plt.show()

    def gridShow(self, train_ds, grid_size=4):
        rand_inds = np.random.randint(0, len(train_ds), grid_size)

        x_grid = [train_ds[i][0] for i in rand_inds]
        y_grid = [train_ds[i][1] for i in rand_inds]

        x_grid = make_grid(x_grid, nrow=4, padding=4)
        plt.rcParams['figure.figsize'] = (10., 5)
        self.showImage(x_grid, y_grid)

    def transformType(self, Datatype="train"):
        if Datatype.lower() == "train":
            print("transform ==> train")
            return transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(45),
                transforms.RandomResizedCrop(96, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
                transforms.ToTensor()
            ])
        else:
            print("transform ==> validation")
            return transforms.Compose([transforms.ToTensor()])

    def dataLoaderGenerator(self, ds, batchSize=32, train=True):
        return DataLoader(ds, batch_size=batchSize, shuffle=train, drop_last=True)

if __name__ == '__main__':
    path = "../../Data/histopathologic-cancer/"
    df = pd.read_csv(os.path.join(path, "test_labels.csv"))

    chacker = CheckandDraw()
    chacker.drawPictures(df, 4, 4, info=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    data = HistoCancerDataset(data_dir=path, transform=transform, data_type="train")

    train_ds, val_ds = data.train_test_split(data)

    # remember everytime you want to use train or val loader, you should reset the transform.
    train_ds.dataset.transform = data.transformType(Datatype="train")
    train_loader = train_ds.dataset.dataLoaderGenerator(train_ds, batchSize=32, train=True)

    val_ds.dataset.transform = data.transformType(Datatype="validation")
    val_loader = val_ds.dataset.dataLoaderGenerator(val_ds, batchSize=64, train=False)

