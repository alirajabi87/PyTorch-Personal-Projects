"""
ResNet model for Object detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        C_in, H_in, W_in = params["input_shape"]
        init_f = params["initial_filters"]
        num_outputs = params["num_outputs"]

        self.conv1 = nn.Conv2d(C_in, init_f, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(init_f + C_in, init_f * 2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(3 * init_f + C_in, init_f * 4, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(7 * init_f + C_in, init_f * 8, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(15 * init_f + C_in, init_f * 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * init_f, num_outputs)

    def forward(self, x):
        identity = F.avg_pool2d(x, 4, 4)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.cat((x, identity), dim=1)

        identity = F.avg_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.cat((x, identity), dim=1)

        identity = F.avg_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.cat((x, identity), dim=1)

        identity = F.avg_pool2d(x, 2, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.cat((x, identity), dim=1)

        x = F.relu(self.conv5(x))
        x = F.adaptive_avg_pool2d(x, 1)

        x = nn.Flatten()(x)
        # x = x.view(x.size(0), -1)
        # x = x.reshape(x.size(0), -1)

        return self.fc1(x)


if __name__ == '__main__':
    params = dict(input_shape=(3, 256, 256),
                  initial_filters=16,
                  num_outputs=2)

    model = Net(params)
    print(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    # summary(model, params["input_shape"])

    test = torch.rand((1, 3, 256, 256), dtype=torch.float32).to(device)
    print(model(test))
