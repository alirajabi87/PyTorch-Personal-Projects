import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
from Unet_Segmentation import Unet

from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    chack_accuracy,
    save_prediction_as_imgs,
)

# Hyperparameters

LEARNING_RATE = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
Num_epochs = 3
Num_workers = 1
IMG_H = 160
IMG_W = 240
PIN_MEMORY = True
LOAD_MODEL = True

TRAIN_IMG_PATH = "../Data/Image_segmentation/train/"
TRAIN_MASK_PATH = "../Data/Image_segmentation/train_masks/"

VAL_IMG_PATH = "../Data/Image_segmentation/test/"
VAL_MASK_PATH = "../Data/Image_segmentation/test_masks"


def train(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_ind, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.float().unsqueeze(1).to(device)  # //TODO: delete float()

        with torch.cuda.amp.autocast():  # //TODO: Do this without this
            prediction = model(data)
            loss = loss_fn(prediction, targets)

        # Backward
        optimizer.zero_grad()  # //TODO: Do this in the old way
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())  # //TODO: Why?


def main():
    train_transforms = A.Compose(
        [
            A.Resize(height=IMG_H, width=IMG_W),
            A.Rotate(limit=30, p=0.8),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0., 0., 0.],
                std=[1., 1., 1.],
                max_pixel_value=255.,
            ),
            ToTensorV2(),
        ]
    )
    val_transforms = A.Compose(
        [
            A.Resize(height=IMG_H, width=IMG_W),
            A.Normalize(
                mean=[0., 0., 0.],
                std=[1., 1., 1.],
                max_pixel_value=255.,
            ),
            ToTensorV2(),
        ]
    )

    model = Unet(in_channel=3, out_channel=1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()  # because the model output is Linear not sigmoid
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_PATH,
        TRAIN_MASK_PATH,
        VAL_IMG_PATH,
        VAL_MASK_PATH,
        BATCH_SIZE,
        train_transforms,
        val_transforms,
        Num_workers,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pt"), model)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(Num_epochs):
        train(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        chack_point = dict(
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict(),
        )
        save_checkpoint(chack_point)

        # check accuracy
        chack_accuracy(val_loader, model, device=device)

        # print some example in folder
        save_prediction_as_imgs(val_loader, model)


if __name__ == '__main__':
    main()
