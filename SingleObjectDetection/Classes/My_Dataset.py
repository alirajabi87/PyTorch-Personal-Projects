import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms.functional as TF

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set()

from PIL import Image, ImageDraw
from sklearn.model_selection import ShuffleSplit


def readDataset(path_base):
    df = pd.read_excel(os.path.join(path_base, "Fovea_location.xlsx"), index_col="ID")

    # df.replace(0, None, inplace=True,)
    # df.dropna(inplace=True)
    # print(df.head())
    # print(df.size)
    # AorN = [imn[0] for imn in df.imgName]
    # sns.scatterplot(x='Fovea_X', y='Fovea_Y', data=df, hue=AorN)
    # plt.show()
    return df


def random_IDs(dataset, ncols=3, nrows=2):
    plt.rcParams['figure.figsize'] = (15., 9.)
    plt.subplots_adjust(wspace=0, hspace=0.3)
    imgName = dataset['imgName']
    ids = dataset.index
    rndIds = np.random.choice(ids, nrows * ncols)
    # print(rndIds)
    return rndIds


def loadImg(dataset, id, path):
    imgname = dataset["imgName"]
    if imgname[id][0] == "A":
        prefix = "AMD"
    else:
        prefix = "Non-AMD"

    Fullpath = os.path.join(path, prefix, imgname[id])
    img = Image.open(Fullpath)
    x = dataset["Fovea_X"][id]
    y = dataset["Fovea_Y"][id]
    label = (x, y)
    return img, label


def show_img_label(img, label, w_h=(50, 50), thickness=2, showplus=None):
    w, h = w_h
    cx, cy = label
    draw = ImageDraw.Draw(img)
    draw.rectangle(((cx - w / 2, cy - h / 2), (cx + w / 2, cy + h / 2)),
                   outline="green", width=thickness)
    plt.imshow(np.asarray(img))
    if showplus:
        x, y = label
        plt.plot(x, y, 'b+', markersize=20)



def resize_img_label(image, label=(0., 0.), target_size=(256, 256)):
    w_orig, h_orig = image.size
    w_targ, h_targ = target_size
    cx, cy = label
    img_new = TF.resize(image, target_size)
    label_new = (cx / w_orig * w_targ, cy / h_orig * h_targ)
    return img_new, label_new


def random_hflip(img, label):
    w, h = img.size
    x, y = label

    img = TF.hflip(img)
    label = (w - x, y)
    return img, label


def random_vflip(img, label):
    w, h = img.size
    x, y = label

    label = (x, h - y)
    img = TF.vflip(img)

    return img, label


def random_shift(img, label, max_translate=(0.2, 0.2)):
    w, h = img.size
    max_t_w, max_t_h = max_translate
    cx, cy = label

    trans_coef = np.random.rand() * 2 - 1
    w_t = int(trans_coef * max_t_w * w)
    h_t = int(trans_coef * max_t_h * h)

    img = TF.affine(img, translate=[w_t, h_t], shear=[0], angle=0, scale=1)
    label = (cx + w_t, cy + h_t)
    return img, label


def scale_label(a, b):
    return [ai / bi for ai, bi in zip(a, b)]


def rescale_label(a, b):
    return [bi * ai for ai, bi in zip(a, b)]


def transformer(img, label, params):
    img, label = resize_img_label(img, label, params["target_size"])

    if np.random.random() < params["p_hflip"]:
        img, label = random_hflip(img, label)

    if np.random.random() < params["p_vflip"]:
        img, label = random_vflip(img, label)

    if np.random.random() < params["p_shift"]:
        img, label = random_shift(img, label, params["max_translate"])

    if np.random.random() < params["p_brightness"]:
        brightness_factor = 1 + (np.random.rand() * 2 - 1) * params["brightness_factor"]
        img = TF.adjust_brightness(img, brightness_factor)

    if np.random.random() < params["p_contrast"]:
        contrast_factor = 1 + (np.random.rand() * 2 - 1) * params["contrast_factor"]
        img = TF.adjust_contrast(img, contrast_factor)

    if np.random.random() < params["p_gamma"]:
        gamma_factor = 1 + (np.random.rand() * 2 - 1) * params["gamma_factor"]
        img = TF.adjust_gamma(img, gamma_factor)

    if params["scale_label"]:
        label = scale_label(label, params["target_size"])

    img = TF.to_tensor(img)
    return img, label


class AMD_Dataset(Dataset):
    def __init__(self, BASE_DIR, transform, params):
        super(AMD_Dataset, self).__init__()
        path2labels = os.path.join(BASE_DIR, "train")
        df = readDataset(path2labels)
        self.labels = df[["Fovea_X", "Fovea_Y"]].values
        self.imgName = df["imgName"]
        self.ids = df.index
        self.fullPath2img = [0] * len(self.ids)
        for id_ in self.ids:
            if self.imgName[id_][0] == "A":
                prefix = "AMD"
            else:
                prefix = "Non-AMD"
            self.fullPath2img[id_ - 1] = os.path.join(BASE_DIR, "training400", prefix, self.imgName[id_])
        self.transform = transform
        self.params = params

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        img = Image.open(self.fullPath2img[item])
        label = self.labels[item]
        img, label = self.transform(img, label, self.params)
        return img, label

def dataLoader(BASE_DIR):
    train_params = dict(target_size=(256, 256),
                        p_hflip=0.5,
                        p_vflip=0.5,
                        p_shift=0.5,
                        max_translate=(0.2, 0.2),
                        p_brightness=0.5,
                        brightness_factor=0.2,
                        p_contrast=0.5,
                        contrast_factor=0.2,
                        p_gamma=0.5,
                        gamma_factor=0.2,
                        scale_label=True)

    val_params = dict(target_size=(256, 256),
                      p_hflip=0.0,
                      p_vflip=0.0,
                      p_shift=0.0,
                      p_brightness=0.0,
                      p_contrast=0.0,
                      p_gamma=0.0,
                      gamma_factor=0.0,
                      scale_label=True)

    amd_ds1 = AMD_Dataset(BASE_DIR, transform=transformer, params=train_params)
    amd_ds2 = AMD_Dataset(BASE_DIR, transform=transformer, params=val_params)

    sss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    indices = range(len(amd_ds1))
    for train_ind, val_ind in sss.split(indices):
        continue
        # print(len(train_ind))
        # print("-" * 50)
        # print(len(val_ind))

    train_ds = Subset(amd_ds1, train_ind)
    val_ds = Subset(amd_ds2, val_ind)

    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=16, shuffle=False)
    return train_dl, val_dl, train_ds, val_ds

if __name__ == '__main__':
    # nrows, ncols = 2, 3
    # df = readDataset(BASE_DIR)
    # rndIds = random_IDs(df, nrows=nrows, ncols=ncols)
    # path = "../../Data/AMD/Training400"
    # # for i, ids in enumerate(rndIds):
    # #     img, label=loadImg(df, ids, path)
    # #     print(np.asarray(img).shape)
    # #     print(img.size, label)
    # #     plt.subplot(nrows, ncols, i+1)
    # #     show_img_label(img, label, w_h=(150, 150), thickness=20)
    # #     plt.title(df["imgName"][ids])
    # id_ = np.random.choice(df.index, 1)
    # img, label = loadImg(df, id_[0], path)
    # plt.subplots_adjust(hspace=0, wspace=0.2)
    # plt.subplot(1, 2, 1)
    # show_img_label(img.copy(), label, w_h=(350, 350), thickness=20)

    train_dl, vl_dl, train_ds, val_ds = dataLoader()

    img, label = next(iter(train_dl))
    print(label)
    show_img_label(TF.to_pil_image(img[0]),
                   rescale_label((label[0][0], label[1][0]), (256, 256)), showplus=True)
    plt.show()
    print("-"*50)
    for img_b, label_b in train_dl:
        print(img_b.shape, img_b.dtype)
        print()
        # Convert to tensor
        label_b = torch.stack(label_b, 1)
        label_b = label_b.type(torch.float32)

        print(label_b)
        print(label_b.shape, label_b.dtype)
        break
