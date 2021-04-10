from SingleObjectDetection.Classes.utils import *
from SingleObjectDetection.Classes.Model import Net
from SingleObjectDetection.Classes.My_Dataset import *

import torch
import torchvision
import numpy as np


def main():
    BASE_DIR = "../Data/AMD/"
    train_dl, val_dl, train_ds, val_ds = dataLoader(BASE_DIR)
    params_model = dict(input_shape=(3, 256, 256),
                        initial_filters=16,
                        num_outputs=2)
    model = Net(params_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.load_state_dict(torch.load("./model/weights.pt"))
    model.eval()
    # print(model)
    loss_func = torch.nn.SmoothL1Loss(reduction="sum")
    with torch.no_grad():
        metric, loss = loss_epoch(model=model, loss_func=loss_func, dataset_dl=val_dl, device=device, opt=None,
                                  sanity_check=False)

    print(loss, metric)

    rndInds = np.random.randint(len(val_ds), size=10)
    print(rndInds)

    plt.rcParams['figure.figsize'] = (15, 10)
    plt.subplots_adjust(wspace=0, hspace=0.15)

    for i, rndi in enumerate(rndInds):
        img, label_true = val_ds[rndi]
        h, w = img.shape[1:]

        with torch.no_grad():
            label_pred = model(img.unsqueeze(0).to(device))[0].cpu()
        plt.subplot(2, 3, i+1)
        show_tensor2labels(img, label_true, label_pred)

        label_true_bb = cxcy2bbox(torch.tensor(label_true).unsqueeze(0))
        label_pred_bb = cxcy2bbox(label_pred.unsqueeze(0))
        iou = torchvision.ops.box_iou(label_pred_bb, label_true_bb)
        plt.title(f"{iou.item()*100:.2f}%")

        if i > 4:
            break

    plt.show()


    # Deploying model on individual image

    path = "../Data/AMD/train/Fovea_location.xlsx"

    label_df = pd.read_excel(path, index_col="ID")

    img, label = loadImg(label_df, 1, path="../Data/AMD/Training400/")

    img, label = resize_img_label(img, label,target_size=(256, 256))

    img = TF.to_tensor(img)
    label = scale_label(label, (256, 256))

    with torch.no_grad():
        label_pred = model(img.unsqueeze(0).to(device))[0].cpu()

    show_tensor2labels(img, label, label_pred)
    plt.show()

if __name__ == '__main__':
    main()
