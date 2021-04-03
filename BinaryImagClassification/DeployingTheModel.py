
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from BinaryImagClassification.classes.Model import Net
from BinaryImagClassification.classes.utils import *
from BinaryImagClassification.classes.dataset import HistoCancerDataset, CheckandDraw
from sklearn.metrics import accuracy_score

import os
import time
import numpy as np
import matplotlib.pyplot as plt

def deploy_model(model, dataset, device, num_classes=2, sanity_check=False):
    len_data = len(dataset)
    y_out = torch.zeros(len_data, num_classes)
    y_gt = np.zeros((len_data), dtype="uint8")
    model = model.to(device)
    elapsed_times = []
    with torch.no_grad():
        for i in range(len_data):
            x, y = dataset[i]
            y_gt[i] = y
            start = time.time()
            y_out[i] = model(x.unsqueeze(0).to(device))
            elapsed_times.append(time.time()-start)

            if sanity_check:
                break
    inference_time = np.mean(elapsed_times)*1000
    print(f" average inference time per image on {device}: {inference_time:.2f} ms")
    return y_out.numpy(), y_gt


def main():
    params_model = dict(input_shape=(3, 96, 96), initial_filter=16, num_hidden_fc=100,
                        num_classes=2, dropout_rate=0.25)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = Net(params_model)
    # summary(model, params_model["input_shape"], device=device.type)

    if os.path.exists("./models/weights.pt"):
        print("==> loading the model...")
        model.load_state_dict(torch.load("./models/weights.pt"))
        model.eval()

    path = "../Data/histopathologic-cancer/"
    transform = transforms.Compose([transforms.ToTensor()])

    data = HistoCancerDataset(data_dir=path, transform=transform, data_type="train")
    data.transform = data.transformType(Datatype="validation")

    _, val_data = data.train_test_split(data)
    # data.transform = data.transformType(Datatype="validation")

    print(len(val_data))
    y_out, y_gt = deploy_model(model, dataset=val_data, device=device, sanity_check=False)
    print(y_out.shape, y_gt.shape)

    y_pred = np.argmax(y_out, axis=1)
    print(y_pred.shape)
    acc = accuracy_score(y_pred=y_pred, y_true=y_gt)
    print(f"accuracy: {acc*100:.2f}%")

    grid_size = 4
    # rand_inds = np.random.randint(0, len(data), grid_size)

    x_grid_test = [data[i][0] for i in range(grid_size)]
    y_grid_test = [data[i][1] for i in range(grid_size)]

    x_grid_test = make_grid(x_grid_test, nrow=4, padding=2)
    plt.rcParams['figure.figsize'] = (10., 5)
    data.showImage(x_grid_test, y_grid_test)


if __name__ == '__main__':
    main()
