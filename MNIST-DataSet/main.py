# Uncomment if you have problem of downloading the MNIST Dataset
# from six.moves import urllib
# opener = urllib.request.build_opener()
# opener.addheaders = [('User-agent', 'Mozilla/5.0')]
# urllib.request.install_opener(opener)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0), std=(1))])

train_data = datasets.MNIST(root='Data/MNIST/',
                            download=True,
                            train=True,
                            transform=transform)

test_data = datasets.MNIST(root='Data/MNIST/',
                           download=True,
                           train=False,
                           transform=transform)

# image, label = train_data[0]
# print(image.max(), image.min())
# plt.imshow(image.reshape((28, 28)))
# plt.title(label)
# plt.show()

# load data to DataLoader

train_loader = DataLoader(train_data, batch_size=10, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=20, shuffle=False, pin_memory=True)


# from torchvision.utils import make_grid
#
# np.set_printoptions(formatter=dict(int=lambda x: f'{x:4}'))
#
# for image, label in train_loader:
#     break

# # for first 12 images
# print('Label: ', label[:12].numpy())
# im = make_grid(image[:12], nrow=12)
# plt.figure(figsize=(12, 6))
# plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
# plt.title(label[:12].numpy())
# plt.show()


class Model(nn.Module):
    def __init__(self, input_size=784, output=10, layers=[256, 128], DropOut=0.4):
        super(Model, self).__init__()

        layerList = []

        layerList.append(nn.Conv2d(in_channels=1,  # number of color channels
                                   out_channels=16,  # number of filters
                                   kernel_size=(3, 3),
                                   stride=1,
                                   padding_mode='zeros'))
        layerList.append(nn.BatchNorm2d(16))
        layerList.append(nn.ReLU(inplace=True))
        layerList.append(nn.MaxPool2d((2, 2)))

        layerList.append(nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1))
        layerList.append(nn.BatchNorm2d(32))
        layerList.append(nn.ReLU(inplace=True))
        layerList.append(nn.MaxPool2d((2, 2)))

        layerList.append(nn.Flatten())
        layerList.append(nn.Dropout(DropOut, inplace=True))
        input_size = 800  # (((28-2) //2)-2) //2
        for i in layers:
            layerList.append(nn.Linear(input_size, i))
            layerList.append(nn.ReLU(inplace=True))
            layerList.append(nn.Dropout(DropOut, inplace=True))
            input_size = i
        layerList.append(nn.Linear(layers[-1], output))
        # layerList.append(F.log_softmax(output, dim=1))
        self.layers = nn.Sequential(*layerList)

    def forward(self, X):
        return F.log_softmax(self.layers(X), dim=1)


def main():
    model = Model().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(model.eval())
    epochs = 6
    train_losses = []
    test_losses = []
    train_correct = []
    test_correct = []

    for i in range(epochs):
        trn_crr = 0
        tst_crr = 0

        for b, (x_train, y_train) in enumerate(train_loader):
            x_train = x_train.cuda()
            y_train = y_train.cuda()
            b += 1
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)

            # predicted = torch.max(y_pred.data, 1)[1]
            predicted = y_pred.argmax(axis=1)

            batch_corr = (predicted == y_train).sum()
            trn_crr += batch_corr

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not b % 100:
                accuracy = trn_crr.item() / b * 10
                print(f"Epoch: {i + 1}, Batch: {b:6}, loss: {loss.item():6.4f}, accuracy: {accuracy:6.2f}")

        train_losses.append(loss)
        train_correct.append(trn_crr)

        with torch.no_grad():
            for b, (X_test, y_test) in enumerate(test_loader):
                X_test = X_test.cuda()
                y_test = y_test.cuda()
                y_val = model(X_test)
                # pred_test = torch.max(y_val.data, 1)[1]
                pred_test = y_val.argmax(axis=1)
                tst_crr += (pred_test == y_test).sum()

        loss = criterion(y_val, y_test)
        test_losses.append(loss)
        test_correct.append(tst_crr)


    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.show()

    plt.plot([t / 600 for t in train_correct], label='training accuracy')
    plt.plot([t / 100 for t in test_correct], label='test_accuracy')
    plt.legend()
    plt.show()

    test_load_all = DataLoader(test_data, batch_size=10000, shuffle=False)
    with torch.no_grad():
        for X_test, y_test in test_load_all:
            X_test = X_test.cuda()
            y_test = y_test.cuda()
            y_val = model(X_test)
            predicted = y_val.argmax(axis=1)
    predicted = predicted.cpu()
    y_test = y_test.cpu()
    print(confusion_matrix(predicted.view(-1), y_test.view(-1)))
    print(classification_report(predicted.view(-1), y_test.view(-1)))

if __name__ == '__main__':
    main()
