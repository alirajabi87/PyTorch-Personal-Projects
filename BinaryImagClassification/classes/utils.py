import torch
import copy
from copy import deepcopy
from .dataset import HistoCancerDataset
from torchvision.transforms import transforms

import matplotlib.pyplot as plt
import os

def get_lr(opt):
    for param in opt.param_groups:
        return param['lr']


def metrics_batch(output, target):
    pred = output.argmax(dim=1, keepdim=True)
    return pred.eq(target.view_as(pred)).sum().item()


def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)

    with torch.no_grad():
        metric_b = metrics_batch(output, target)

    if opt:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), metric_b


def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for xb, yb in dataset_dl:
        xb = xb.to(device)
        yb = yb.to(device)

        pred = model(xb)
        loss_b, metric_b = loss_batch(loss_func=loss_func, output=pred, target=yb, opt=opt)

        running_loss += loss_b
        running_metric += metric_b

        if sanity_check:
            break

    loss = running_loss / float(len_data)
    metric = running_metric / float(len_data)

    return loss, metric


def train_val(model, param):
    num_epochs = param["num_epochs"]
    loss_func = param["loss_func"]
    opt = param["optimizer"]
    train_dl = param["train_dl"]
    val_dl = param["val_dl"]
    sanity_check = param["sanity_check"]
    lr_scheduler = param["lr_scheduler"]
    path2weights = param["path2weights"]



    # history of loss values in each epoch
    loss_history = dict(train=[], val=[])
    metric_history = dict(train=[], val=[])

    # Continue the training

    if os.path.exists(path2weights):
        print(" ==> loading model ...")
        model.load_state_dict(torch.load(path2weights))

    best_model_weight = deepcopy(model.state_dict())

    # initializing loss value to inf
    best_loss = float('inf')

    path = "../Data/histopathologic-cancer/"
    transform = transforms.Compose([transforms.ToTensor()])
    data = HistoCancerDataset(data_dir=path, transform=transform, data_type="train")

    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print(f'Epoch {epoch+1}/{num_epochs}, current lr= {current_lr}')

        model.train()
        data.transform = data.transformType("train")
        train_loss, train_metric = loss_epoch(model=model,
                                              loss_func=loss_func,
                                              dataset_dl=train_dl,
                                              sanity_check=sanity_check,
                                              opt=opt)
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)

        model.eval()

        with torch.no_grad():
            data.transform = data.transformType("validation")
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl, sanity_check)
            loss_history["val"].append(val_loss)
            metric_history["val"].append(val_metric)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_weight = deepcopy(model.state_dict())
                torch.save(model.state_dict(), path2weights)
                print("copied best model")

            lr_scheduler.step(val_loss)

            if current_lr != get_lr(opt):
                print("Loading best model weights!")
                model.load_state_dict(best_model_weight)

            print(f"tain loss: {train_loss}, val loss: {val_loss}, accuracy: {100*val_metric}")
            print("-"*10)

    model.load_state_dict(best_model_weight)
    return model, loss_history, metric_history

def plotResults(title, num_epochs, df, xLabel, yLabel, train_Label, val_Label):
    plt.title(title)
    plt.plot(range(1, num_epochs + 1), df["train"], label=train_Label)
    plt.plot(range(1, num_epochs + 1), df["val"], label=val_Label)
    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    plt.legend()
    plt.show()