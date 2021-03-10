import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0), std=(1))])
train_data = datasets.CIFAR10('Data/CIFAR10/',
                              download=True,
                              train=True,
                              transform=transform)
test_data = datasets.CIFAR10('Data/CIFAR10/',
                             download=True,
                             train=False,
                             transform=transform)

train_loader = DataLoader(train_data,
                          shuffle=True,
                          batch_size=10,
                          pin_memory=True)
test_loader = DataLoader(test_data,
                         shuffle=False,
                         batch_size=20,
                         pin_memory=True)
labels = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat',
          4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}


# # For checking the dataset
# data = enumerate(train_loader)
# ind, (image, label) = next(data)
# print(image.shape)
# for i in range(1):
#     plt.imshow(image[i][0])
#     plt.title(labels[label[i].item()])
#     plt.show()

class Model(nn.Module):
    def __init__(self, output=10, layers=[256, 256, 128], DropeOut=0.4):
        super(Model, self).__init__()
        layerList = []
        # First Conv
        layerList.append(nn.Conv2d(3, 32, 3, 1))
        layerList.append(nn.BatchNorm2d(32))
        layerList.append(nn.ReLU(inplace=True))
        layerList.append(nn.MaxPool2d((2, 2)))
        # Second Conv
        layerList.append(nn.Conv2d(32, 64, 3, 1))
        layerList.append(nn.BatchNorm2d(64))
        layerList.append(nn.ReLU(inplace=True))
        layerList.append(nn.MaxPool2d((2, 2)))
        # Flatten
        layerList.append(nn.Flatten())
        # Dropout
        layerList.append(nn.Dropout(DropeOut))

        i_input = (((32 - 2) // 2 - 2) // 2) ** 2 * 64
        # Dense Layers
        for i in layers:
            layerList.append(nn.Linear(i_input, i))
            layerList.append(nn.ReLU(inplace=True))
            layerList.append(nn.Dropout(DropeOut))
            i_input = i
        layerList.append(nn.Linear(layers[-1], output))
        self.layers = nn.Sequential(*layerList)

    def forward(self, X):
        return F.log_softmax(self.layers(X), dim=1)


def PrintOutput(epoch, accuracy, loss, b=None, val_loss=None, verbose=2):
    if verbose == 0:
        pass
    elif verbose == 1:
        print(f"Epoch: {epoch:3}, Batch: {b:5}, Loss: {loss:6.4f}, Accuracy: {accuracy:6.2f}")
    elif verbose == 2:
        print(f"Epoch: {epoch:3}, Loss: {loss:6.4f}, Val_loss: {val_loss:6.4f}, Accuracy: {accuracy:6.2f}")


def main():
    model = Model().cuda()
    # print(model.eval())
    Epochs = 5
    train_losses = []
    test_losses = []
    train_accuracy = []
    test_accuracy = []

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    import time
    startTime = time.time()
    for i in range(Epochs):
        loss = 0
        trn_crt = 0
        test_crt = 0
        i += 1
        verbose = 2
        for b, (X_train, y_train) in enumerate(train_loader):
            X_train = X_train.cuda()
            y_train = y_train.cuda()
            b += 1
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)

            predicted = y_pred.argmax(axis=1)
            trn_crt += (predicted == y_train).sum()

            if not b % 100 and verbose == 1:
                accuracy = trn_crt.item() / b * 10
                PrintOutput(epoch=i, accuracy=accuracy, loss=loss, b=b, verbose=verbose)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_losses.append(loss)
        train_accuracy.append(trn_crt)

        with torch.no_grad():
            for b, (X_test, y_test) in enumerate(test_loader):
                X_test = X_test.cuda()
                y_test = y_test.cuda()
                test_pred = model(X_test)
                test_loss = criterion(test_pred, y_test)
                test_crt += (test_pred.argmax(axis=1) == y_test).sum()
        test_losses.append(test_loss)
        test_accuracy.append(test_crt)

        if verbose == 2:
            accuracy = trn_crt.item() / b
            PrintOutput(epoch=i, accuracy=accuracy, loss=loss, val_loss=test_loss, verbose=verbose)

    print(f"Duration: {(time.time() - startTime) / 60} min")
    plt.plot(train_losses, label="train_loss")
    plt.plot(test_losses, label="test_loss")
    plt.title("loss")
    plt.legend()
    plt.show()

    plt.plot([t / 600 for t in train_accuracy], label='training accuracy')
    plt.plot([t / 100 for t in test_accuracy], label='test_accuracy')
    plt.legend()
    plt.show()

    # Validation:
    test_all = DataLoader(test_data, shuffle=False, batch_size=10000)
    with torch.no_grad():
        val_loss = []
        for X_val, y_val in test_all:
            X_val = X_val.cuda()
            y_val = y_val.cuda()
            prediction = model(X_val)
            loss = criterion(prediction, y_val)
            prediction = prediction.argmax(axis=1)

            val_loss.append(loss)
            prediction = prediction.cpu()
            y_val = y_val.cpu()
        print(classification_report(prediction.view(-1), y_val.view(-1)))
        print(f"Validation loss: {val_loss}")
        df = pd.DataFrame(
            confusion_matrix(y_pred=prediction.view(-1), y_true=y_val.view(-1)),
            list(labels.values()),
            list(labels.values())
        )
        import seaborn as sns
        sns.heatmap(df, annot=True, fmt="d", cmap='BuGn')
        plt.xlabel("predictions")
        plt.ylabel("labels")
        plt.show()


if __name__ == '__main__':
    main()
