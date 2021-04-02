import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import numpy as np


class Encoder(nn.Module):
    def __init__(self, C_in, H_in, W_in, init_filter):
        super(Encoder, self).__init__()
        self.filters = [C_in, init_filter, init_filter * 2, init_filter * 4, init_filter * 8]
        # , kernel_size=3, padding=1, stride=2
        self.convBlocks = [self.convBlock(in_c, out_c, kernel_size=3, bias=False)
                           for in_c, out_c in zip(self.filters, self.filters[1:])]
        for i in range(len(self.convBlocks)):
            H, W = self.findConv2dOutputShape(H_in, W_in, self.convBlocks[0][0])
            H_in, W_in = H, W

        self.H, self.W = H, W

        self.convOutput = H * W * 8 * init_filter

        self.encoder = nn.Sequential(*self.convBlocks)

    def forward(self, x):
        return self.encoder(x)

    def convBlock(self, in_c, out_c, BN=True, *args, **kwargs):
        layers = nn.ModuleList()
        layers.append(nn.Conv2d(in_c, out_c, *args, **kwargs))
        if BN:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(2, 2))
        return nn.Sequential(*layers)

    def findConv2dOutputShape(self, H_in, W_in, conv, pool=2):
        # get parameters
        kernel_size = conv.kernel_size
        stride = conv.stride
        padding = conv.padding
        dilation = conv.dilation

        H_out = np.floor((H_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
        W_out = np.floor((W_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)

        if pool:
            H_out /= pool
            W_out /= pool

        return int(H_out), int(W_out)


class Decoder(nn.Module):
    def __init__(self, num_flatten, num_hidden_fc, num_classes, dropout):
        super(Decoder, self).__init__()
        self.num_flatten = num_flatten
        self.num_hidden_fc = num_hidden_fc
        self.num_classes = num_classes
        self.dropout = dropout
        self.decoder = self.decoreBloc()

    def forward(self, x):
        return F.log_softmax(self.decoder(x), dim=1)

    def decoreBloc(self):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_flatten, self.num_hidden_fc),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.num_hidden_fc, self.num_classes)
        )


class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        C_in, H_in, W_in = params["input_shape"]
        init_filter = params["initial_filter"]
        num_hidden_fc = params["num_hidden_fc"]
        num_classes = params["num_classes"]
        dropout = params["dropout_rate"]

        self.encoder = Encoder(C_in, H_in, W_in, init_filter)
        self.decoder = Decoder(self.encoder.convOutput, num_hidden_fc, num_classes, dropout)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def main():
    params_model = dict(input_shape=(3, 96, 96), initial_filter=8, num_hidden_fc=100,
                        num_classes=2, dropout_rate=0.25)

    print(params_model)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = Net(params_model).to(device)
    # print(model.eval())
    summary(model, params_model["input_shape"], device=device.type)

    img_test = torch.rand(size=(1, 3, 96, 96)).to(device)
    print(model(img_test))

if __name__ == '__main__':
    main()
