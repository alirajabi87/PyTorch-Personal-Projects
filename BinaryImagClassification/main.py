import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import matplotlib.pyplot as plt
from torchvision.transforms import transforms

from BinaryImagClassification.classes.dataset import HistoCancerDataset
from BinaryImagClassification.classes.Model import Net
from BinaryImagClassification.classes.utils import *


def main():
    path = "../Data/histopathologic-cancer/"
    transform = transforms.Compose([transforms.ToTensor()])

    data = HistoCancerDataset(data_dir=path, transform=transform, data_type="train")

    train_ds, val_ds = data.train_test_split(data)

    # remember everytime you want to use train or val loader, you should reset the transform.
    train_ds.dataset.transform = data.transformType(Datatype="train")
    train_loader = train_ds.dataset.dataLoaderGenerator(train_ds, batchSize=32, train=True)

    val_ds.dataset.transform = data.transformType(Datatype="validation")
    val_loader = val_ds.dataset.dataLoaderGenerator(val_ds, batchSize=64, train=False)

    # Model

    params_model = dict(input_shape=(3, 96, 96), initial_filter=16, num_hidden_fc=100,
                        num_classes=2, dropout_rate=0.25)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = Net(params_model).to(device)
    summary(model, params_model["input_shape"], device=device.type)

    # Loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                        patience=15, verbose=1)

    params_train = dict(
        num_epochs=60,
        loss_func=criterion,
        optimizer=optimizer,
        train_dl=train_loader,
        val_dl=val_loader,
        sanity_check=False,
        lr_scheduler=lr_scheduler,
        path2weights="./models/weights.pt"
    )

    model, loss_history, metric_history = train_val(model, params_train)
    num_epochs = params_train["num_epochs"]

    plotResults("Train_val Loss", num_epochs, loss_history,
                "Loss", "epoch", "train loss", "val loss")

    plotResults("Train_val accuracy", num_epochs, metric_history,
                "Accuracy", "epoch", "train acc", "val acc")


if __name__ == '__main__':
    main()
