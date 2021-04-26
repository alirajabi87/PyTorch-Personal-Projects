import torch
import torch.nn.functional as F
import torchvision
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import time


def dice_loss(pred, target, smooth=1e-5):
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

    dice = 2.0 * (intersection + smooth) / (union + smooth)
    loss = 1.0 - dice
    return loss.sum(), dice.sum()


def loss_func(pred, target):
    # bce = torch.nn.BCEWithLogitsLoss(reduction='sum')(pred, target)
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='sum')

    pred = torch.sigmoid(pred)
    dlv, _ = dice_loss(pred, target)

    loss = bce + dlv
    return loss


def metric_batch(pred, target):
    pred = torch.sigmoid(pred)

    _, metric = dice_loss(pred, target)

    return metric

def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)

    with torch.no_grad():
        metric = metric_batch(output, target)

    if opt:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), metric.item()

def loss_epoch(model, loss_func, dataset_dl, device, sanity_check=False, opt=None):

    len_data = len(dataset_dl.dataset)
    epoch_loss = 0.0
    epoch_metric = 0.0

    for x_b, y_b in dataset_dl:
        x_b = x_b.to(device)
        y_b = y_b.to(device)
        pred = model(x_b)

        loss_b, metric_b = loss_batch(loss_func=loss_func, output=pred, target=y_b, opt=opt)

        epoch_loss += loss_b

        if metric_b:
            epoch_metric += metric_b

        if sanity_check:
            break
    loss = epoch_loss / float(len_data)
    metric = epoch_metric / float(len_data)

    return loss, metric

def get_lr(opt):
    for params in opt.param_groups:
        return params['lr']

def train_val(params):
    model = params["model"]
    train_dl = params["train_dl"]
    val_dl = params["val_dl"]
    num_epochs = params["num_epochs"]
    device = params["device"]
    sanity_check = params["sanity_check"]
    opt = params["optimizer"]
    lr_scheduler = params["lr_scheduler"]
    path2weights = params["path2weights"]

    history = dict(train_loss=[], train_acc=[], val_loss=[], val_acc=[])

    best_loss = float("inf")
    best_model_weights = deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        startTime = time.time()
        current_lr = get_lr(opt)
        print(f"epoch: {epoch+1}/{num_epochs}, current learning rate: {current_lr:.2e}")
        model.train()
        loss, metric = loss_epoch(model=model, loss_func=loss_func, dataset_dl=train_dl, device=device, sanity_check=sanity_check, opt=opt)

        history["train_loss"].append(loss)
        history["train_acc"].append(metric)

        with torch.no_grad():
            model.eval()
            val_loss, val_metric = loss_epoch(model=model, loss_func=loss_func, dataset_dl=val_dl, device=device)

            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_metric)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_weights = deepcopy(model.state_dict())
                print(" ==> saving the best model weights...")
                torch.save(model.state_dict(), path2weights)

        lr_scheduler.step(val_loss)

        if current_lr != get_lr(opt):
            print("==> loading the best model weights for new learning rate ...")
            model.load_state_dict(best_model_weights)
        print(f"train loss: {loss:6.4f}, val loss: {val_loss:6.4f}, train accuracy: {metric * 100:4.2f} %, val accuracy: {val_metric * 100:4.2f} %")
        print(f"epoch time: {(time.time()-startTime)/60:.2f} min")
        print("-"*50)

    model.load_state_dict(best_model_weights)
    return model, history

def plotResults(title, num_epochs, data1, data2, xLabel, yLabel, train_Label, val_Label):
    plt.title(title)
    plt.plot(range(1, num_epochs + 1), data1, label=train_Label)
    plt.plot(range(1, num_epochs + 1), data2, label=val_Label)
    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    plt.legend()
    plt.show()