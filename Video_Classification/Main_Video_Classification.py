from Video_Classification.Classes.Utils import *
from Video_Classification.Classes.Model import get_model
from Video_Classification.Classes.MyDataset import dataLoader

import pandas as pd

def main():
    model_type = 'rnn'
    # model_type = "3dcnn"

    if model_type == "rnn":
        model = get_model(7, 'rnn').to(device)
    else:
        model = get_model(7, '3dcnn').to(device)

    train_dl, val_dl = dataLoader(4, model_type)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5,
                                                              patience=5, mode="min", verbose=1)

    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    train_params = dict(num_epochs=20, loss_fn=loss_fn,
                        train_dl=train_dl,
                        val_dl=val_dl,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        sanity_check=False,
                        path2weights="./model/weights.pt", )
    model, history = train_val(model, train_params)

    df = pd.DataFrame.from_dict(history)
    print(df.head())
    plotResults("loss", num_epochs=train_params["num_epochs"],
                data1=df["train_loss"], data2=df["val_loss"],
                xLabel="epoch", yLabel="Loss",
                train_Label="train_loss", val_Label="val_loss")


if __name__ == '__main__':
    main()
