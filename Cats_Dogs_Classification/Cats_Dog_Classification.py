import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import time

warnings.filterwarnings('ignore')


class Encoder(nn.Module):
    def __init__(self, encoder_size):
        super(Encoder, self).__init__()
        self.encoder_size = encoder_size

        enc_block = [self.encoderBlock(in_c, out_c, kernel_size=3, stride=1, padding=1)
                     for in_c, out_c in zip(self.encoder_size, self.encoder_size[1:])]

        self.encoder = nn.Sequential(*enc_block)

    def forward(self, x):
        return self.encoder(x)

    def encoderBlock(self, in_c, out_c, *args, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, *args, **kwargs),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )


class Decoder(nn.Module):
    def __init__(self, decoder_size, n_classes):
        super(Decoder, self).__init__()

        self.decoder_size = decoder_size

        dec_block = [self.decoderBlock(in_c, out_c)
                     for in_c, out_c in zip(self.decoder_size, self.decoder_size[1:])]

        self.decoder = nn.Sequential(*dec_block,
                                     nn.Linear(self.decoder_size[-1], n_classes))

    def forward(self, x):
        return self.decoder(x)

    def decoderBlock(self, in_c, out_c):
        return nn.Sequential(
            nn.Linear(in_c, out_c),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )


class MyClassificationClass(nn.Module):
    def __init__(self, in_channels, encoder_hidden_layers, decoder_hidden_layers, n_classes, image_size):
        super(MyClassificationClass, self).__init__()

        self.encoder_size = [in_channels, *encoder_hidden_layers]
        self.decoder_size = [41472, *decoder_hidden_layers]
        # print(self.encoder_size)
        # print(self.decoder_size)

        self.encoder = Encoder(self.encoder_size)
        self.decoder = Decoder(self.decoder_size, n_classes)
        self.Flat_Drop = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.Flat_Drop(x)
        # print(x.shape)
        x = self.decoder(x)
        return F.log_softmax(x, dim=1)


def imageLoader(path, train_data=True, batch_size=32, **kwargs):
    if train_data:
        transform = transforms.Compose([
            transforms.Resize(150),
            transforms.CenterCrop(150),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.465, 0.406), std=(0.229, 0.224, 0.225))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(150),
            transforms.CenterCrop(150),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.465, 0.406), std=(0.229, 0.224, 0.225))
        ])
    data = datasets.ImageFolder(root=path, transform=transform)
    loader = DataLoader(dataset=data,
                        batch_size=batch_size,
                        shuffle=train_data,
                        drop_last=True,
                        pin_memory=True,
                        **kwargs)
    class_name = data.classes
    return loader, class_name


def main():
    TrainPath = "../OpenCVDeep/DATA/CATS_DOGS/train"
    TestPath = "../OpenCVDeep/DATA/CATS_DOGS/test"

    Training_Images = glob(TrainPath + "/*/*.jp*g")
    Test_Images = glob(TestPath + "/*/*.jp*g")
    # print(len(Training_Images), len(Test_Images))

    train_loader, classes_name = imageLoader(TrainPath, train_data=True, batch_size=10)
    test_loader, _ = imageLoader(TestPath, train_data=False, batch_size=10)

    # check the images

    images, labels = next(iter(train_loader))
    im = make_grid(images, nrow=5)
    inv_transform = transforms.Normalize(
        mean=(-0.485 / 0.229, -0.465 / 0.224, -0.406 / 0.225),
        std=(1 / 0.229, 1 / 0.224, 1 / 0.225))
    im = inv_transform(im)
    plt.figure(figsize=(12, 4))
    plt.imshow(np.transpose(im, (1, 2, 0)))
    plt.show()

    model = MyClassificationClass(in_channels=3, encoder_hidden_layers=[32, 64, 128],
                                  decoder_hidden_layers=[256, 128, 64],
                                  n_classes=2, image_size=(150, 150))
    if torch.cuda.is_available():
        model.cuda()
    # print(model)
    EPOCHS = 15
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []
    startTime = time.time()
    for i in range(EPOCHS):
        train_loss = 0
        train_crt = 0
        val_loss = 0
        val_crt = 0

        for b, (X_train, y_train) in enumerate(train_loader):
            X_train, y_train = X_train.cuda(), y_train.cuda()
            y_pred = model(X_train)
            train_loss = criterion(y_pred, y_train)
            train_crt += (y_pred.argmax(1) == y_train).sum()
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        train_losses.append(train_loss)
        train_acc.append(train_crt)
        del X_train
        del y_train
        torch.cuda.empty_cache()

        with torch.no_grad():
            for X_val, y_val in test_loader:
                X_val, y_val = X_val.cuda(), y_val.cuda()
                y_pred = model(X_val)
                val_loss = criterion(y_pred, y_val)
                val_crt += (y_pred.argmax(1) == y_val).sum()
            val_losses.append(val_loss)
            val_acc.append(val_crt)
            del X_val
            del y_val
            torch.cuda.empty_cache()
        print(f"epoch: {i + 1:3}, train_loss: {train_loss:6.3f}, Val_loss: {val_loss:6.3f}")

    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.legend()
    plt.show()

    plt.plot([t / len(Training_Images) * 100 for t in train_acc], label='training accuracy')
    plt.plot([t / len(Test_Images) * 100 for t in val_acc], label='test_accuracy')
    plt.legend()
    plt.show()

    validation_loader, _ = imageLoader(TestPath, train_data=False, batch_size=100)
    with torch.no_grad():
        for val_data, val_label in validation_loader:
            val_data, val_label = val_data.cuda(), val_label.cuda()
            y_pred = model(val_data)
            val_loss = criterion(y_pred, val_label)
            prediction = y_pred.argmax(axis=1)

            prediction = prediction.cpu()
            val_label = val_label.cpu()
    print(classification_report(prediction.view(-1), val_label.view(-1)))
    df = pd.DataFrame(
        confusion_matrix(y_pred=prediction.view(-1), y_true=val_label.view(-1)),
        index=classes_name,
        columns=classes_name
    )
    sns.heatmap(df, annot=True, fmt="d", cmap='BuGn')
    plt.xlabel("predictions")
    plt.ylabel("labels")
    plt.show()
    torch.save(model.state_dict(), 'Cats_Dogs_Classifier.pt')
    print(f"Time: {(time.time() - startTime) / 60} min")

    # #Single image Classification
    # model.load_state_dict(torch.load("Cats_Dogs_classifier.pt"))
    # img_index = 1000
    # transform = transforms.Compose([
    #     transforms.Resize(150),
    #     transforms.CenterCrop(150),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.485, 0.465, 0.406), std=(0.229, 0.224, 0.225))
    # ])
    # test_data = datasets.ImageFolder(root=TestPath, transform=transform)
    # im = inv_transform(test_data[img_index][0])
    # plt.imshow(np.transpose(im, (1,2,0)))
    # plt.show()
    # model.eval()
    # with torch.no_grad():
    #     new_pred = model(test_data[img_index][0].view(1, 3, 150, 150).cuda()).argmax()
    # print(classes_name[new_pred.item()])


if __name__ == '__main__':
    main()
