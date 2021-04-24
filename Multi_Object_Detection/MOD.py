import os
import torch
import torchvision

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from Multi_Object_Detection.classes.utils import *
from Multi_Object_Detection.classes.Model import DarkNet
from Multi_Object_Detection.classes.Dataset import dataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    path = "../Data/coco"

    train_dl, val_dl, _, _ = dataLoader(path)
    model = DarkNet(os.path.join(path, "yolov3.cfg")).to(device)
    # print(model)
    xb,  yb, _ = next(iter(train_dl))
    model(xb.to(device))
    del xb, yb
    torch.cuda.empty_cache()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", patience=20, verbose=1)

    path2model = "./model"
    if not os.path.exists(path2model):
        os.mkdir(path2model)
    scaled_anchor = [model.module_list[82][0].scaled_anchors,
                     model.module_list[94][0].scaled_anchors,
                     model.module_list[106][0].scaled_anchors]

    mse_loss = torch.nn.MSELoss(reduction="sum")
    bce_loss = torch.nn.BCELoss(reduction="sum")

    params_loss = dict(scaled_anchors=scaled_anchor,
                       ignore_thres=0.5,
                       mse_loss=mse_loss,
                       bce_loss=bce_loss,
                       num_yolos=3,
                       num_anchors=3,
                       obj_scale=1,
                       noobj_scale=100)
    path2weights = os.path.join(path2model, "weights.pt")
    params_train = dict(num_epochs=2,
                        optimizer=optimizer,
                        params_loss=params_loss,
                        train_dl=train_dl,
                        val_dl=val_dl,
                        sanity_check=False,
                        lr_scheduler=lr_scheduler,
                        path2weights=path2weights)

    model, loss_history = train_val(model, params_train)

    plotResults(title="loss", num_epochs=params_train["num_epochs"], df=loss_history,
                xLabel="epoch", yLabel="loss", train_Label="train_loss", val_Label="val_loss")


if __name__ == '__main__':
    main()