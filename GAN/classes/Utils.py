import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
DCGAN ==> process
1. Get a batch of real images with the target labels set to 1.
2. Generate a batch of fake images using the generator with the target labels set to 0.
3. Feed the mini-batches to the discriminator and compute the loss and gradients.
4. Update the discriminator parameters using the gradients.
5. Generate a batch of fake images using the generator with the target labels set to 1.
6. Feed the fake mini-batch to the discriminator and compute the loss and gradients.
7. Update the generator only based on gradients.
8. Repeat from step 1.
"""

H, W = 64, 64
REAL_LABEL = 1.0
FAKE_LABEL = 0.0
NZ = 100

def train_gen_dis(model_dis, model_gen, params):
    num_epochs = params["num_epochs"]
    train_dl = params["train_dl"]
    opt_gen = params["optimizer_gen"]
    opt_dis = params["optimizer_dis"]
    loss_func = params["loss_func"]
    path2weights = params["path2weights"]
    path_gen = os.path.join(path2weights, "weights_gen.pt")
    path_dis = os.path.join(path2weights, "weights_dis.pt")

    loss_history = dict(gen=[], dis=[])
    batch_count = 0
    for epoch in range(num_epochs):
        for xb, yb in train_dl:
            ba_si = xb.size(0)
            model_dis.zero_grad()
            # 1 & 2
            xb = xb.to(device)
            yb = torch.full((ba_si,), REAL_LABEL, device=device)
            # 3
            out_dis = model_dis(xb)
            loss_r = loss_func(out_dis, yb)
            loss_r.backward()
            # 4
            noise = torch.randn(ba_si, NZ, 1, 1, device=device)
            out_gen = model_gen(noise)
            out_dis = model_dis(out_gen.detach())
            yb.fill_(FAKE_LABEL)
            loss_f = loss_func(out_dis, yb)
            loss_f.backward()
            loss_diss = loss_r + loss_f
            opt_dis.step()
            # 5
            model_gen.zero_grad()
            yb.fill_(REAL_LABEL)
            # 6
            out_dis = model_dis(out_gen)
            # 7
            loss_gen = loss_func(out_dis, yb)
            loss_gen.backward()
            opt_gen.step()


            loss_history["gen"].append(loss_gen.item())
            loss_history["dis"].append(loss_diss.item())
            batch_count += 1
            if not batch_count % 100:
                print(f"epoch: {epoch}, loss_gen: {loss_gen.item():.3f}, loss_dis: {loss_diss.item():.3f}")
    torch.save(model_gen.state_dict(), path_gen)
    torch.save(model_dis.state_dict(), path_dis)
    return loss_history



def plotResults(title, num_epochs, data1, data2, xLabel, yLabel, train_Label, val_Label):
    plt.title(title)
    plt.plot(range(1, num_epochs + 1), data1, label=train_Label)
    plt.plot(range(1, num_epochs + 1), data2, label=val_Label)
    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    plt.legend()
    plt.show()