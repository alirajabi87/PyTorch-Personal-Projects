import torch
import torch.nn as nn
import torchvision
from Video_Classification.Classes.Utils import *


class Resnet18Rnn(nn.Module):
    def __init__(self, params_model):
        super(Resnet18Rnn, self).__init__()
        num_classes = params_model["num_classes"]
        dr_rate = params_model["dr_rate"]
        pretrained = params_model["pretrained"]
        rnn_hidden_size = params_model["rnn_hidden_size"]
        rnn_num_layers = params_model["rnn_num_layers"]

        baseModel = torchvision.models.resnet18(pretrained=pretrained)
        num_features = baseModel.fc.in_features
        baseModel.fc = Identity()

        self.baseModel = baseModel
        self.dropOut = nn.Dropout(dr_rate)
        self.rnn = nn.LSTM(num_features, rnn_hidden_size, rnn_num_layers)
        self.fc1 = nn.Linear(rnn_hidden_size, num_classes)

    def forward(self, x):
        b_z, ts, c, h, w = x.shape
        ii = 0

        y = self.baseModel((x[:, ii]))
        output, (hn, cn) = self.rnn(y.unsqueeze(1))

        for ii in range(1, ts):
            y = self.baseModel((x[:, ii]))
            output, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))
        output = self.dropOut(output[:, -1])
        output = self.fc1(output)
        return output


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def get_model(num_classes, model_type="rnn"):
    if model_type.lower() == 'rnn':
        params_model = dict(num_classes=num_classes, dr_rate=0.1, pretrained=True,
                            rnn_hidden_size=100, rnn_num_layers=1)
        model = Resnet18Rnn(params_model)
    else:
        model = torchvision.models.video.r3d_18(pretrained=True, progress=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    return model


if __name__ == '__main__':
    x = torch.zeros(1, 16, 3, 224, 224)
    x3d = torch.zeros(1, 3, 16, 112, 112)

    model_rnn = get_model(7, 'rnn').to(device)
    model_3dcnn = get_model(7, '3dcnn').to(device)

    with torch.no_grad():
        y = model_rnn(x.to(device))
        y_3d = model_3dcnn(x3d.to(device))

    print(f"model rnn: {y}")
    print(f"model 3dcnn: {y_3d}")

    print(model_rnn)
    print("-"*100)
    print(model_3dcnn)


