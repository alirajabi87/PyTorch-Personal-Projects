import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DoubleConv, self).__init__()
        self.convBlock = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1,  bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convBlock(x)


class Unet(nn.Module):
    def __init__(self, in_channel=3, out_channel=2, features=[64, 128, 256, 512]):
        super(Unet, self).__init__()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        # Down sampling
        for featur in features:
            self.downs.append(
                DoubleConv(in_channel, featur)
            )
            in_channel = featur

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleNeck = DoubleConv(features[-1], features[-1] * 2)

        # Up sampling
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.output = nn.Conv2d(features[0], out_channel, kernel_size=1)

    def forward(self, x):
        skip_lists = []

        for module in self.downs:
            x = module(x)
            skip_lists.append(x)
            x = self.pool(x)

        x = self.bottleNeck(x)

        skip_lists = skip_lists[::-1]

        for ind in range(0, len(self.ups), 2):

            x = self.ups[ind](x)
            if x.shape is not skip_lists[ind//2].shape:
                x = TF.resize(x, size=skip_lists[ind//2].shape[2:])

            x = torch.cat((x , skip_lists[ind//2]), dim=1)

            x = self.ups[ind+1](x)
        return self.output(x)


def test():
    a = torch.randn((3, 1, 161, 161))
    model = Unet(in_channel=1, out_channel=1)
    # print(model)
    predict = model(a)

    print(predict.size())
    print(a.size())

    assert predict.size() == a.size()

if __name__ == '__main__':
    test()