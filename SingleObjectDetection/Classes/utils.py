import torch
import torchvision
import torchvision.transforms.functional as TF
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image, ImageDraw


def get_lr(opt):
    for param in opt.param_groups:
        return param["lr"]


def cxcy2bbox(cxcy, w=50. / 256, h=50. / 256):
    w_tensor = torch.ones(cxcy.shape[0], 1, device=cxcy.device) * w
    h_tensor = torch.ones(cxcy.shape[0], 1, device=cxcy.device) * h

    cx = cxcy[:, 0].unsqueeze(1)
    cy = cxcy[:, 1].unsqueeze(1)

    boxes = torch.cat((cx, cy, w_tensor, h_tensor), -1)
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,  # X_min, Y_min
                      boxes[:, :2] + boxes[:, 2:] / 2), 1)  # X_max, Y_max


def metric_batch(output, target):
    output = cxcy2bbox(output)
    target = cxcy2bbox(target)

    iou = torchvision.ops.box_iou(output, target)
    return torch.diagonal(iou, 0).sum().item()


def loss_batch(loss_func, output, target, opt=None):
    loss_b = loss_func(output, target)

    with torch.no_grad():
        metric_b = metric_batch(output, target)

    if opt:
        opt.zero_grad()
        loss_b.backward()
        opt.step()
    return loss_b.item(), metric_b


def loss_epoch(model, loss_func, dataset_dl, device, opt=None, sanity_check=True):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)
    for x_b, y_b in dataset_dl:
        y_b = torch.stack(y_b, 1)
        y_b = y_b.type(torch.float32).to(device)

        output = model(x_b.to(device))

        loss_b, metric_b = loss_batch(loss_func, output, y_b, opt)

        running_loss += loss_b
        if metric_b:
            running_metric += metric_b

        if sanity_check:
            break

    metric = running_metric / float(len_data)
    loss = running_loss / float(len_data)

    return metric, loss


def train_val(params):
    model = params["model"]
    num_epochs = params["num_epochs"]
    loss_func = params["loss_func"]
    train_dl = params["train_dl"]
    val_dl = params["val_dl"]
    device = params["device"]
    opt = params["optimizer"]
    sanity_check = params["sanity_check"]
    lr_scheduler = params["lr_scheduler"]
    path2weight = params["path2weight"]

    loss_history = dict(train=[], val=[])
    metric_history = dict(train=[], val=[])

    best_model_weight = deepcopy(model.state_dict())
    best_loss = float("inf")

    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start = time.time()
        current_lr = get_lr(opt)
        print(f"epoch {epoch + 1}/{num_epochs}, current lr: {current_lr}")
        model.train()
        train_metric, train_loss = loss_epoch(model=model,
                                              loss_func=loss_func,
                                              dataset_dl=train_dl,
                                              device=device,
                                              opt=opt,
                                              sanity_check=sanity_check)

        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)

        with torch.no_grad():
            model.eval()

            val_metric, val_loss = loss_epoch(model=model,
                                              loss_func=loss_func,
                                              dataset_dl=val_dl,
                                              device=device,
                                              opt=None,
                                              sanity_check=False)

            loss_history["val"].append(val_loss)
            metric_history["val"].append(val_metric)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_weight = deepcopy(model.state_dict())
            torch.save(model.state_dict(), path2weight)
            print(f"best model weights has been copied in {path2weight}!!")

        lr_scheduler.step(val_loss)

        if current_lr != get_lr(opt):
            print("loading the best model...")
            model.load_state_dict(best_model_weight)

        print(f"train loss: {train_loss:.6f}, train accuracy{train_metric * 100: .2f}%")
        print(f"val loss: {val_loss:.6f}, val accuracy: {val_metric * 100:.2f}%")
        print(f"time passed from start: {(time.time()-start_time) / 60:.2f} min, epoch_duration: {(time.time()-epoch_start)/60} min")
        print("-" * 50)

    model.load_state_dict(best_model_weight)
    return model, loss_history, metric_history, opt


def plotResults(title, num_epochs, df, xLabel, yLabel, train_Label, val_Label):
    plt.title(title)
    plt.plot(range(1, num_epochs + 1), df["train"], label=train_Label)
    plt.plot(range(1, num_epochs + 1), df["val"], label=val_Label)
    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    plt.legend()
    plt.show()


def rescale_label(a, b):
    return [bi * ai for ai, bi in zip(a, b)]


def show_tensor2labels(img, label_true, label_pred, w_h=(50, 50)):
    label_true = rescale_label(label_true, img.shape[1:])
    label_pred = rescale_label(label_pred, img.shape[1:])
    img = TF.to_pil_image(img)

    w, h = w_h

    draw = ImageDraw.Draw(img)
    draw.rectangle(((label_true[0] - w / 2, label_true[1] - h / 2),
                    (label_true[0] + w / 2, label_true[1] + h / 2)), outline="green", width=2)

    draw.rectangle(((label_pred[0] - w / 2, label_pred[1] - h / 2),
                    (label_pred[0] + w / 2, label_pred[1] + h / 2)), outline='red', width=2)

    plt.imshow(img)
