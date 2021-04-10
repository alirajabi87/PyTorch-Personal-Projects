from SingleObjectDetection.Classes.utils import *
from SingleObjectDetection.Classes.Model import Net
from SingleObjectDetection.Classes.My_Dataset import dataLoader
import pandas as pd


def main():
    BASE_DIR = "../Data/AMD/"
    train_dl, val_dl, train_ds, val_ds = dataLoader(BASE_DIR)

    params_model = dict(input_shape=(3, 256, 256),
                        initial_filters=64,
                        num_outputs=2)

    model = Net(params_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    # print(model)

    # define loss function and optimizer
    # we use smoothed-L1 loss function

    loss_func = torch.nn.SmoothL1Loss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                              factor=0.5, patience=10, verbose=1)

    params_train = dict(model=model,
                        num_epochs=1,
                        loss_func=loss_func,
                        train_dl=train_dl,
                        val_dl=val_dl,
                        device=device,
                        optimizer=optimizer,
                        sanity_check=True,
                        lr_scheduler=lr_scheduler,
                        path2weight="./model/weights.pt")

    model, loss_history, metric_history, opt = train_val(params_train)

    plotResults("loss", num_epochs=params_train["num_epochs"], df=loss_history,
                xLabel="epoch", yLabel="loss",
                train_Label="train_loss", val_Label="val_loss")
    plotResults("Accuracy", num_epochs=params_train["num_epochs"], df=metric_history,
                xLabel="epoch", yLabel="accuracy",
                train_Label="train_acc", val_Label="val_acc")

    loss_df = pd.DataFrame.from_dict(loss_history, orient='index', columns=["train_loss", "val_loss"])
    metric_df = pd.DataFrame.from_dict(metric_history, orient='index', columns=["train_acc", "val_acc"])
    df = pd.concat([loss_df, metric_df], axis=1)
    df.to_csv(f"./model/metrics{get_lr(opt)}.csv", index=False)


if __name__ == '__main__':
    main()
