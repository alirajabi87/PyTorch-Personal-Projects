import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from glob import glob


class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img_path = os.path.join(self.img_dir, self.images[item])
        mask_path = os.path.join(self.mask_dir, self.images[item].replace(".jpg", "_mask.gif"))

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255] = 1.0
        # mask /= 255

        if self.transform:
            augmentation = self.transform(image=image, mask=mask)
            image = augmentation["image"]
            mask = augmentation["mask"]

        return image, mask
