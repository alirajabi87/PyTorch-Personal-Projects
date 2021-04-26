import torch
import torchvision
import pandas as pd

from Single_Object_Segmentation.classes.utils import *
from Single_Object_Segmentation.classes.Model import SegNet
from Single_Object_Segmentation.classes.Dataset import dataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    h, w = 128, 192
    BASE_DIR = '../Data/FETAL'
    train_dl, val_dl, train_ds, val_ds = dataLoader(BASE_DIR)

    params_model = dict(input_shape=(1, h, w),
                        initial_filters=16,
                        num_outputs=1)
    model = SegNet(params_model).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=1)

    params_train = dict(model=model,
                        num_epochs=50,
                        device=device,
                        train_dl=train_dl,
                        val_dl=val_dl,
                        lr_scheduler=lr_scheduler,
                        sanity_check=False,
                        optimizer=optimizer,
                        path2weights="./model/weights.pt")

    model, history = train_val(params_train)

    df = pd.DataFrame.from_dict(history)
    # print(df)

    plotResults("loss", params_train["num_epochs"], data1=df["train_loss"], data2=df["val_loss"],
                xLabel="epoch", yLabel="loss", train_Label="tain_loss", val_Label="val_loss")
    plotResults("Accuracy", params_train["num_epochs"], data1=df["train_acc"], data2=df["val_acc"],
                xLabel="epoch", yLabel="Accuracy", train_Label="tain_acc", val_Label="val_acc")



if __name__ == '__main__':
    main()
