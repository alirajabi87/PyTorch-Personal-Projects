import pandas as pd

from GAN.classes.Utils import *
from GAN.classes.Model import *
from GAN.classes.Dataset import *


def main():
    path2data = "../Data"
    train_dl, train_ds = Mydataset(path2data)

    params_gen = dict(nz=100, ngf=64, noc=3)
    model_gen = Generator(params_gen).to(device)
    params_dis = dict(nic=3, ndf=64)
    model_dis = Discriminator(params_dis).to(device)

    loss_func = nn.BCELoss()
    optimizer_dis = torch.optim.Adam(model_dis.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_gen = torch.optim.Adam(model_gen.parameters(), lr=2e-4, betas=(0.5, 0.999))

    params = {"num_epochs": 100,
              "train_dl": train_dl,
              "optimizer_gen": optimizer_gen,
              "optimizer_dis": optimizer_dis,
              "loss_func": loss_func,
              "path2weights": "./model"}
    history = train_gen_dis(model_dis=model_dis, model_gen=model_gen, params=params)
    df = pd.DataFrame.from_dict(history)
    print(len(df["gen"]))
    plotResults("Loss", len(df["gen"]), df["gen"], df["dis"], "epoch", "Loss",
                "Generator", "Discriminator")


if __name__ == '__main__':
    main()
