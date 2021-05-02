import matplotlib.pyplot as plt
import torch.utils.data
from torchvision import datasets
from torchvision import transforms

from GAN.classes.Utils import *

path2data = "../../Data"
os.makedirs(path2data, exist_ok=True)


def Mydataset(path2data):
    H, W = 64, 64
    MEAN = (0.5, 0.5, 0.5)
    STD = (0.5, 0.5, 0.5)
    transform = transforms.Compose([
        transforms.Resize((H, W)),
        transforms.CenterCrop((H, W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    train_ds = datasets.STL10(path2data, split='train',
                              download=False,
                              transform=transform)
    train_dl = torch.utils.data.DataLoader(train_ds,
                                           batch_size=16,
                                           shuffle=True,
                                           pin_memory=True)

    return train_dl, train_ds


if __name__ == '__main__':
    Mydataset()
