# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from Multi_Object_Segmentation.classes.Utils import *
from Multi_Object_Segmentation.classes.Dataset import *

from torchvision.models.segmentation import deeplabv3_resnet50
from torchsummary import summary
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loading_model():
    model = deeplabv3_resnet50(pretrained=True, num_classes=21)
    model = model.to(device)
    return model


def main():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    h, w = 520, 520
    path = "../Data/"  # VOC2012"
    train_ds, val_ds = dataset_ds(h, w, mean, std, path=path)
    train_dl, val_dl = dataset_dl(train_ds, val_ds)

    model = loading_model()

    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=10, verbose=1)

    train_params = dict(num_epochs=10,
                        train_dl=train_dl,
                        val_dl=val_dl,
                        sanity_check=True,
                        device=device,
                        loss_func=criterion,
                        opt=optimizer,
                        lr_scheduler=lr_scheduler,
                        path2weights="./model/weights.pt")

    model, history = train_val(model, params=train_params)

    df = pd.DataFrame.from_dict(history)
    # print(df)
    plotResults("loss", train_params["num_epochs"], df["train_loss"], df["val_loss"],
                "epoch", "loss", "train_loss", "val_loss")
    plotResults("Accuracy", train_params["num_epochs"], df["train_acc"], df["val_acc"],
                "epoch", "Accuracy", "train_acc", "val_acc")

if __name__ == '__main__':
    main()
