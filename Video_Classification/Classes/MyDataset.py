import os

import matplotlib.pyplot as plt
import torch

from Video_Classification.Classes.Utils import *
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
import glob
import random

random.seed(2020)
torch.manual_seed(2020)
np.random.seed(2020)

BASE_DIR = "../Data/Videos/"
# BASE_DIR = "../../Data/Videos/"
path_vid = "hmdb51_org"
path_jpg = "hmdb51_jpg"
extension = ".avi"

path2Catgs = os.path.join(BASE_DIR, path_vid)
listofCatgs = os.listdir(path2Catgs)

n_frames = 16

# for root, dirs, files in os.walk(path2Catgs, topdown=False):
#     for name in files:
#         if extension not in name:
#             continue
#         path2vid = os.path.join(root, name)
#         frames, vlen = get_frames(path2vid, n_frames)
#         path2store = path2vid.replace(path_vid, path_jpg)
#         path2store = path2store.replace(extension, "")
#         print(path2store)
#         os.makedirs(path2store, exist_ok=True)
#         storeFrames(frames, path2store)
#     print("-"*50)

path2ajpg = os.path.join(BASE_DIR, path_jpg)

all_vids, all_labels, catgs = get_videos(path2ajpg)

# print(len(all_vids), len(all_labels), len(catgs))
# print(all_vids[:3], all_labels[:3], catgs[:3])

label_dict = {}
ind = 0
for i in catgs:
    label_dict[i] = ind
    ind += 1

# print(label_dict)

num_classes = 7

uniq_ids = [id_ for id_, label in zip(all_vids, all_labels) if label_dict[label] < num_classes]

uniq_label = [label for id_, label in zip(all_vids, all_labels) if label_dict[label] < num_classes]

# Split data
sss = StratifiedShuffleSplit(n_splits=2, test_size=0.1, random_state=0)

train_indx, test_indx = next(sss.split(uniq_ids, uniq_label))

train_ids = [uniq_ids[ind] for ind in train_indx]
train_label = [uniq_label[ind] for ind in train_indx]

test_ids = [uniq_ids[ind] for ind in test_indx]
test_label = [uniq_label[ind] for ind in test_indx]


# print(len(train_ids), len(test_ids))


class myDataset(Dataset):
    def __init__(self, ids, labels, mode, model_type='rnn'):
        super(myDataset, self).__init__()
        self.ids = ids
        self.labels = labels
        # self.transform = transform
        self.mode = mode
        self.model_type = model_type

    def __getitem__(self, item):
        path2img = glob.glob(self.ids[item] + "/*.jpg")
        path2img = path2img[:n_frames]
        label = label_dict[self.labels[item]]
        frames = []
        for p2i in path2img:
            frame = Image.open(p2i)
            frames.append(frame)

        # seed = np.random.randint(1e9)
        #
        # frame_tr = []
        # for frame in frames:
        #     random.seed(seed)
        #     np.random.seed(seed)
        #
        #     frame = self.transform(frame)
        #     frame_tr.append(frame)
        # if len(frame_tr) > 0:
        #     frame_tr = torch.stack(frame_tr)
        frame_tr = transform_frames(frames, model_type='rnn', mode="train")
        return frame_tr, label

    def __len__(self):
        return len(self.ids)


def dataLoader(batch_size=16, model_type='rnn'):
    train_ds = myDataset(ids=train_ids, labels=train_label, mode="train", model_type=model_type)
    val_ds = myDataset(ids=test_ids, labels=test_label, mode="test")

    if model_type == 'rnn':
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              pin_memory=True, collate_fn=collate_fn_rnn)
        val_dl = DataLoader(val_ds, batch_size=2 * batch_size,
                            shuffle=False, pin_memory=True, collate_fn=collate_fn_rnn)
    else:
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              pin_memory=True, collate_fn=collate_fn_3dcnn)
        val_dl = DataLoader(val_ds, batch_size=2 * batch_size, shuffle=False,
                            pin_memory=True, collate_fn=collate_fn_3dcnn)
    return train_dl, val_dl


def collate_fn_rnn(batch):
    imgs_b, label_b = list(zip(*batch))
    imgs_b = [imgs for imgs in imgs_b if len(imgs) > 0]
    label_b = [torch.tensor(l) for l, imgs in zip(label_b, imgs_b) if len(imgs) > 0]
    imgs_tensor = torch.stack(imgs_b)
    label_tensor = torch.stack(label_b)
    return imgs_tensor, label_tensor


def collate_fn_3dcnn(batch):
    imgs_b, label_b = list(zip(*batch))
    imgs_b = [imgs for imgs in imgs_b if len(imgs) > 0]
    label_b = [torch.tensor(l) for l, imgs in zip(label_b, imgs_b) if len(imgs) > 0]
    imgs_tensor = torch.stack(imgs_b)

    imgs_tensor = torch.transpose(imgs_tensor, 2, 1)

    label_tensor = torch.stack(label_b)
    return imgs_tensor, label_tensor


if __name__ == '__main__':
    model_type = 'rnn'
    # model_type = "3dcnn"
    # train_ds = myDataset(ids=train_ids, labels=train_label, mode="train", model_type=model_type)
    # val_ds = myDataset(ids=test_ids, labels=test_label, mode="test")
    # # print(len(train_ds), len(val_ds))
    #
    # imgs, label = train_ds[15]
    #
    # if len(imgs) > 0:
    #     print(imgs.shape, label, torch.min(imgs), torch.max(imgs))
    #
    # plt.figure(figsize=(10, 10))
    # for ii, img in enumerate(imgs[::4]):
    #     plt.subplot(2, 2, (ii + 1))
    #     plt.imshow(denormalize(img, model_type))
    #     plt.title(label)
    #
    # plt.show()

    train_dl, val_dl = dataLoader(16, model_type)

    xb, yb = next(iter(train_dl))
    print(xb.shape, yb.shape)
