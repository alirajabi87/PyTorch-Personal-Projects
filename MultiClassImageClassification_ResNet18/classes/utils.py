import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchvision
from copy import deepcopy


def show(img, y=None, color=True):
    img = np.transpose(img.numpy(), (1, 2, 0))
    plt.imshow(img)
    if y:
        plt.title(f"label: {str(y)}")
    plt.show()


def randomShow(dataset, grid_size=4):
    rand_inds = np.random.randint(0, len(dataset), grid_size)
    x_grid = [dataset[i][0] for i in rand_inds]
    y_grid = [dataset[i][1] for i in rand_inds]

    x_grid = torchvision.utils.make_grid(x_grid, nrow=grid_size, padding=1)
    plt.rcParams['figure.figsize'] = (grid_size * 2 + 2, grid_size * 2 + 2)
    show(x_grid, y_grid)


def plotResults(title, num_epochs, df, xLabel, yLabel, train_Label, val_Label):
    plt.title(title)
    plt.plot(range(1, num_epochs + 1), df["train"], label=train_Label)
    plt.plot(range(1, num_epochs + 1), df["val"], label=val_Label)
    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    plt.legend()
    plt.show()


def get_lr(opt):
    for param_groups in opt.param_groups:
        return param_groups['lr']


def metric_batch(y_out, target):
    y_out = F.log_softmax(y_out, dim=1)
    preds = y_out.argmax(dim=1, keepdim=True)
    corrects = preds.eq(target.view_as(preds)).sum().item()
    return corrects


def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)
    metric_b = metric_batch(output, target)
    if opt:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), metric_b


def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None, device="cuda"):
    epoch_loss = 0.0
    epoch_metric = 0.0
    len_data = len(dataset_dl.dataset)

    for xb, yb in dataset_dl:
        xb = xb.to(device)
        yb = yb.to(device)

        output = model(xb)
        loss_b, metric_b = loss_batch(loss_func=loss_func,
                                      output=output,
                                      target=yb,
                                      opt=opt)
        epoch_loss += loss_b

        if metric_b is not None:
            epoch_metric += metric_b

        if sanity_check:
            break
    loss = epoch_loss / float(len_data)
    metric = epoch_metric / float(len_data)
    return loss, metric


def train_val(model, params):
    num_epochs = params["num_epochs"]
    loss_func = params["loss_func"]
    sanity_check = params["sanity_check"]
    opt = params["optimizer"]
    train_dl = params["train_dl"]
    val_dl = params["val_dl"]
    lr_scheduler = params["lr_scheduler"]
    path2weights = params["path2weights"]
    device = params["device"]

    loss_history = dict(train=[], val=[])
    metric_history = dict(train=[], val=[])

    best_model_wts = deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print(f"Epoch: {epoch + 1}/{num_epochs}, current lr: {current_lr}")

        model.train()
        train_loss, train_metric = loss_epoch(model=model,
                                              loss_func=loss_func,
                                              dataset_dl=train_dl,
                                              sanity_check=sanity_check,
                                              opt=opt,
                                              device=device)
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model=model,
                                              loss_func=loss_func,
                                              dataset_dl=val_dl,
                                              sanity_check=sanity_check,
                                              opt=None,
                                              device=device)
            loss_history["val"].append(val_loss)
            metric_history["val"].append(val_metric)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = deepcopy(model.state_dict())
                torch.save(model.state_dict(), path2weights)
                print("Copied best model")
            lr_scheduler.step()
        print(f"train loss: {train_loss:.5f}, val loss: {val_loss:.5f}, accuracy: {val_metric*100:.2f}%")
        print("-" * 100)

    model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history
