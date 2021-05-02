from torch import nn
import torch.nn.functional as F
from GAN.classes.Utils import *


class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()
        nz = params["nz"]  # The size of the input noise vector
        ngf = params["ngf"]  # Base number of convolutional filters
        noc = params["noc"]  # number of output channels

        self.dconv1 = nn.ConvTranspose2d(nz, ngf * 8,
                                         kernel_size=4, stride=1,
                                         padding=0, bias=False)
        self.BN1 = nn.BatchNorm2d(ngf * 8)

        self.dconv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4,
                                         kernel_size=4, stride=2,
                                         padding=1, bias=False)
        self.BN2 = nn.BatchNorm2d(ngf * 4)

        self.dconv3 = nn.ConvTranspose2d(ngf * 4, ngf * 2,
                                         kernel_size=4, stride=2,
                                         padding=1, bias=False)
        self.BN3 = nn.BatchNorm2d(ngf * 2)

        self.dconv4 = nn.ConvTranspose2d(ngf * 2, ngf * 1,
                                         kernel_size=4, stride=2,
                                         padding=1, bias=False)
        self.BN4 = nn.BatchNorm2d(ngf * 1)

        self.dconv5 = nn.ConvTranspose2d(ngf, noc,
                                         kernel_size=4, stride=2,
                                         padding=1, bias=False)

    def forward(self, x):
        x = F.relu(self.BN1(self.dconv1(x)))
        x = F.relu(self.BN2(self.dconv2(x)))
        x = F.relu(self.BN3(self.dconv3(x)))
        x = F.relu(self.BN4(self.dconv4(x)))
        return torch.tanh(self.dconv5(x))


class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()
        nic = params["nic"]  # input channel
        ndf = params["ndf"]  # Base number of convolutional filters

        self.conv1 = nn.Conv2d(nic, ndf, kernel_size=4, stride=2, padding=1, bias=False)
        self.BN1 = nn.BatchNorm2d(ndf)

        self.conv2 = nn.Conv2d(ndf * 1, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.BN2 = nn.BatchNorm2d(ndf * 2)

        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.BN3 = nn.BatchNorm2d(ndf * 4)

        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.BN4 = nn.BatchNorm2d(ndf * 8)

        self.conv5 = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
        x = F.leaky_relu(self.BN2(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.BN3(self.conv3(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.BN4(self.conv4(x)), 0.2, inplace=True)
        return torch.sigmoid(self.conv5(x)).view(-1)


def initialize_weights(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


if __name__ == '__main__':
    params_gen = dict(nz=100, ngf=64, noc=3)
    model_gen = Generator(params_gen).to(device)
    print(model_gen)

    with torch.no_grad():
        y = model_gen(torch.zeros(1, 100, 1, 1, device=device))

    print(y.shape)

    params_gen = dict(nic=3, ndf=64)
    model_dis = Discriminator(params_gen).to(device)
    print(model_dis)

    with torch.no_grad():
        y = model_dis(torch.zeros(1, 3, H, W, device=device))

    print(y.shape)

    model_gen.apply(initialize_weights)
    model_dis.apply(initialize_weights)
