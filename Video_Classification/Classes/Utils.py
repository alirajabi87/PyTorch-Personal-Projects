import os

import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2 as cv
from PIL import Image

from tqdm import tqdm
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_videos(path2jpgs):
    listOfCats = os.listdir(path2jpgs)
    ids = []
    labels = []

    for catg in listOfCats:
        path2catg = os.path.join(path2jpgs, catg)
        listOfSubCats = os.listdir(path2catg)
        path2subCats = [os.path.join(path2catg, subC) for subC in listOfSubCats]
        ids.extend(path2subCats)
        labels.extend([catg] * len(listOfSubCats))
    return ids, labels, listOfCats


def denormalize(x_, model_type):
    if model_type.lower() == "rnn":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.43216, 0.394666, 0.37645]
        std = [0.22803, 0.22145, 0.216989]
    x = x_.clone()
    for i in range(3):
        x[i] = x[i] * std[i] + mean[i]
    return to_pil_image(x)


def get_frames(filename, n_frame=1):
    frames = []
    v_cap = cv.VideoCapture(filename)
    v_len = int(v_cap.get(cv.CAP_PROP_FRAME_COUNT))
    frame_list = np.linspace(0, v_len - 1, n_frame + 1, dtype=np.int16)

    for fn in range(v_len):
        success, frame = v_cap.read()
        if not success:
            continue
        if (fn in frame_list):
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frames.append(frame)
    v_cap.release()
    return frames, v_len


def storeFrames(frames, path2store):
    for ii, frame in enumerate(frames):
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        path2img = os.path.join(path2store, f"frame{str(ii)}.jpg")
        cv.imwrite(path2img, frame)


def transform_frames(frames, model_type='rnn', mode="train"):
    if model_type.lower() == "rnn":
        h, w = 224, 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        h, w = 112, 112
        mean = [0.43216, 0.394666, 0.37645]
        std = [0.22803, 0.22145, 0.216989]

    if mode == "train":
        transformer = transforms.Compose([
            transforms.Resize((h, w)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        transformer = transforms.Compose([
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    frames_tr = []
    for frame in frames:
        frames_tr.append(transformer(frame))

    if len(frames_tr)>0:
        frames_tr = torch.stack(frames_tr)

    if model_type == "3dcnn":
        frames_tr = torch.transpose(frames_tr, 1, 0)

    # imgs_tensor = imgs_tensor.unsqueeze(0)
    return frames_tr


def plotResults(title, num_epochs, data1, data2, xLabel, yLabel, train_Label, val_Label):
    plt.title(title)
    plt.plot(range(1, num_epochs + 1), data1, label=train_Label)
    plt.plot(range(1, num_epochs + 1), data2, label=val_Label)
    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    plt.legend()
    plt.show()


# train and Validation

def get_lr(opt):
    for param in opt.param_groups:
        return param["lr"]


def metric_batch(outputs, targets):
    pred = outputs.argmax(dim=1, keepdim=True)
    corrects = pred.eq(targets.view_as(pred)).sum()
    return corrects.item()


def batch_loss(loss_func, outputs, targets, opt=None):
    loss = loss_func(outputs, targets)
    with torch.no_grad():
        metric = metric_batch(outputs, targets)

    if opt:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), metric


def epoch_loss(model, loss_func, dataset_dl, sanity_check=True, opt=None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)

    for xb, yb in dataset_dl:
        yb = yb.to(device)
        output = model(xb.to(device))

        loss_b, metric_b = batch_loss(loss_func, output, yb, opt)

        running_loss += loss_b

        if metric_b:
            running_metric += metric_b

        if sanity_check:
            break

    loss = running_loss / float(len_data)
    metric = running_metric / float(len_data)

    return loss, metric


def train_val(model, params):
    num_epochs = params["num_epochs"]
    loss_fn = params["loss_fn"]
    train_dl = params["train_dl"]
    val_dl = params["val_dl"]
    opt = params["optimizer"]
    lr_scheduler = params["lr_scheduler"]
    sanity_check = params["sanity_check"]
    path2weights = params["path2weights"]

    best_model_weights = deepcopy(model.state_dict())
    best_loss = float('inf')

    history = dict(train_loss=[], train_metric=[], val_loss=[], val_metric=[])

    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print(f"epoch {epoch + 1}/{num_epochs}, current lr: {current_lr}")

        model.train()
        train_loss, train_metric = epoch_loss(model=model,
                                              loss_func=loss_fn,
                                              dataset_dl=train_dl,
                                              sanity_check=sanity_check,
                                              opt=opt)
        history["train_loss"].append(train_loss)
        history["train_metric"].append(train_metric)

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = epoch_loss(model=model,
                                              dataset_dl=val_dl,
                                              loss_func=loss_fn,
                                              sanity_check=sanity_check)

        history["val_loss"].append(val_loss)
        history["val_metric"].append(val_metric)

        if val_loss < best_loss:
            best_loss = val_loss
            print(" ==> Saving the best model weights ...")
            best_model_weights = deepcopy(model.state_dict())
            torch.save(model.state_dict(), path2weights)

        lr_scheduler.step(val_loss)

        if current_lr != get_lr(opt):
            print("==> loading the best model weights ...")
            model.load_state_dict(best_model_weights)

        print(f"train loss: {train_loss:.3f}, train acc: {train_metric * 100} % "
              f"val loss: {val_loss:.3f}, val acc: {val_metric * 100} %")
        print("-" * 50)

    model.load_state_dict(best_model_weights)

    return model, history
