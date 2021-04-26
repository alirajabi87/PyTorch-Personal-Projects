import torch
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.transforms.functional import to_pil_image
import numpy as np
import matplotlib.pyplot as plt

from skimage.segmentation import mark_boundaries
import time
from copy import deepcopy

NUM_CLASSES = 21
COLORS = np.random.randint(0, 2, size=(NUM_CLASSES + 1, 3), dtype="uint8")
CUDA_LAUNCH_BLOCKING=1

def show_img_target(img, target):
    if torch.is_tensor(img):
        img = to_pil_image(img)
        target = target.numpy()

    for ll in range(NUM_CLASSES):
        mask = (target == ll)
        img = mark_boundaries(np.array(img),
                              mask,
                              outline_color=COLORS[ll],
                              color=COLORS[ll])
        plt.imshow(img)


def show_img_mask_boundaries(img, mask, mean, std):
    img_r = renormalize(img, mean, std)
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(to_pil_image(img_r))

    plt.subplot(1, 3, 2)
    plt.imshow(mask)

    plt.subplot(1, 3, 3)
    show_img_target(img_r, mask)
    plt.show()


def renormalize(x, mean, std):
    x_r = x.clone()
    for c, (mean_c, std_c) in enumerate(zip(mean, std)):
        x_r[c] *= std_c
        x_r[c] += mean_c
    return x_r


def get_lr(opt):
    for param in opt.param_groups:
        return param['lr']


def metric_batch(pred, target):
    pred = pred.cpu()
    pred = torch.argmax(pred, dim=1)
    metric = pred.eq(target.cpu().view_as(pred)).sum().item()
    return metric


def loss_batch(loss_func, pred, target, opt=None):
    loss = loss_func(pred, target)
    with torch.no_grad():
        metric = metric_batch(pred, target)

    if opt:
        opt.zero_grad()
        loss.backward()
        opt.step()
        # opt.zero_grad()
        # loss.backward()
        # opt.step()

    return loss.item(), metric


def loss_epoch(model, loss_func, dataset_dl, device, sanity_check=False, opt=None):
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    len_data = len(dataset_dl.dataset)
    # with torch.autograd.set_detect_anomaly(True):
    for xb, yb in dataset_dl:
        pred = model(xb.to(device))['out']

        loss_b, metric_b = loss_batch(loss_func=loss_func, pred=pred,
                                      target=yb.to(device), opt=opt)
        print(loss_b, metric_b)
        if metric_b:
            epoch_accuracy += metric_b

        epoch_loss += loss_b

        if sanity_check:
            break

    return epoch_loss / float(len_data), epoch_accuracy / float(len_data)


def train_val(model, params):
    num_epochs = params["num_epochs"]
    train_dl = params["train_dl"]
    val_dl = params["val_dl"]
    sanity_check = params["sanity_check"]
    device = params["device"]
    loss_func = params["loss_func"]
    opt = params["opt"]
    lr_scheduler = params["lr_scheduler"]
    path2weights = params["path2weights"]

    history = dict(train_loss=[], train_acc=[], val_loss=[], val_acc=[])

    best_model_weights = deepcopy(model.state_dict())
    best_loss = float("inf")


    for epoch in range(num_epochs):
        startTime = time.time()
        current_lr = get_lr(opt)
        print(f"epoch ==> {epoch+1}/{num_epochs}, current learning rate: {current_lr}")

        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func=loss_func, dataset_dl=train_dl,
                                              device=device, sanity_check=sanity_check, opt=opt)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_metric)

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func=loss_func, dataset_dl=val_dl,
                                              device=device)
            if val_loss < best_loss:
                best_loss = val_loss
                print("==> saving the best model weights ...")
                best_model_weights = deepcopy(model.state_dict())
                torch.save(model.state_dict(), path2weights)

            lr_scheduler.step(val_loss)
            if current_lr != get_lr(opt):
                print("==> loading the best weights from previous all time best ...")
                model.load_state_dict(best_model_weights)

            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_metric)
        print(f"loss: {train_loss:6.4f}, val_loss: {val_loss:6.4f}, accuracy: {train_metric/100:.2f}%, val_acc:{val_metric/100:.2f}%")
        print(f"time duration fot this epoch: {(time.time()-startTime)/60:.2f} min")
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


if __name__ == '__main__':
    from Multi_Object_Segmentation.classes.Dataset import dataset_dl, dataset_ds

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    h, w = 520, 520
    path = "../../Data/"  # VOC2012"
    train_ds, val_ds = dataset_ds(h, w, mean, std, path=path)
    train_dl, val_dl = dataset_dl(train_ds, val_ds)

    img, mask = next(iter(train_dl))
    model = deeplabv3_resnet50(pretrained=True, num_classes=21).cuda()
    pred = model(img.cuda())["out"]
    print(pred.shape, mask.shape)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    loss, metric = loss_batch(criterion, pred, mask.cuda())
    print(loss, metric)