import os
import torch
from torch import nn
import torchvision
from torchvision.models import resnet18, vgg19
from torchsummary import summary
from MultiClassImageClassification.classes.utils import *
from MultiClassImageClassification.classes.dataset import *


def main():
    # Data section
    train_dl, val_dl, test_dl = dataGenerator()

    # Loading the model
    model = resnet18(pretrained=True)
    print(model)

    num_classes = 10
    num_filters = model.fc.in_features  # for Resnet-18

    print(num_filters)
    model.fc = nn.Linear(num_filters, num_classes)  # for Resnet-18

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    summary(model, input_size=(3, 224, 224), device=device.type)

    # Defining the loss function
    criterion = nn.CrossEntropyLoss(reduction="sum")

    # Defining the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=5, eta_min=1e-6,
    )

    os.makedirs("./model/", exist_ok=True)
    params = dict(num_epochs=100,
                  optimizer=optimizer,
                  loss_func=criterion,
                  lr_scheduler=lr_scheduler,
                  train_dl=train_dl,
                  val_dl=val_dl,
                  sanity_check=False,
                  path2weights="./model/weights.pt",
                  device=device)
    model, loss_history, metric_history = train_val(model, params)

    plotResults("loss", params["num_epochs"], loss_history, "epoch", "loss", "train_loss", "val_loss")
    plotResults("accuracy", params["num_epochs"], metric_history, "epoch", "accuracy", "train acc", "val_acc")


if __name__ == '__main__':
    main()
