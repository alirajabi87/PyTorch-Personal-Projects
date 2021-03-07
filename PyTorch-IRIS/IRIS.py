import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../../PYTORCH_NOTEBOOKS/Data/iris.csv')


def plot(DataFram):
    df = DataFram
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
    fig.tight_layout()

    plots = [(0, 1), (2, 3), (0, 2), (1, 3)]
    colors = ['b', 'r', 'g']
    labels = ['Iris Setosa', 'Iris Virginica', 'Iris Versicolor']

    for i, ax in enumerate(axes.flat):
        for j in range(3):
            x = df.columns[plots[i][0]]
            y = df.columns[plots[i][1]]
            ax.scatter(df[df['target'] == j][x], df[df['target'] == j][y], color=colors[j], label=labels[j])
            ax.set(xlabel=x, ylabel=y)

    fig.legend(labels, loc=3, bbox_to_anchor=(1.0, 0.85))
    plt.show()


X = df.drop('target', axis=1).values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)

X_train = torch.FloatTensor(X_train).cuda()
X_test = torch.FloatTensor(X_test).cuda()
y_train = torch.LongTensor(y_train).cuda()
y_test = torch.LongTensor(y_test).cuda()


trainloader = DataLoader(X_train, batch_size=60, shuffle=True, pin_memory=True)
testLoader = DataLoader(X_train, batch_size=60, shuffle=True, pin_memory=True)

# Model Creation
class Model(nn.Module):

    def __init__(self, in_features=4, h1=16, h2=16,  out_features=3):
        # Input layer (4 features) --> hidden layer 1 (N) --> hidden layer 2 (N)--> output (3 classes)
        super().__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=h1)
        self.fc2 = nn.Linear(in_features=h1, out_features=h2)
        self.out = nn.Linear(in_features=h2, out_features=out_features)


    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        # x = F.softmax(self.out(x))
        x = self.out(x)
        return x




torch.manual_seed(32)
model = Model().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
EPOCHS = 150
losses = []

for i in range(EPOCHS):
    i += 1
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss)

    if not i % 20:
        print(f" Epoch: {i}, loss: {loss}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), 'iris.pt') # to save entire model just put model
new_model = Model()
new_model.load_state_dict(torch.load('iris.pt'))
print(new_model.eval())


plt.plot(losses, label='losses per epoch')
plt.show()

with torch.no_grad():
    pred = model.forward(X_test)
    loss = criterion(pred, y_test)

print(loss)
from sklearn.metrics import confusion_matrix, classification_report
print(pred.cpu().numpy().shape)
pred = pred.cpu()
pred = np.argmax(pred.numpy(), axis=-1)
print(confusion_matrix(y_pred=pred, y_true=y_test.cpu()))
print(classification_report(y_true=y_test.cpu(), y_pred=pred))


mystery_iris = torch.tensor([5.6, 3.7, 2.2, 0.5])
with torch.no_grad():
    pred = np.argmax(new_model(mystery_iris))
    pred2 = new_model(mystery_iris).argmax()
    print(f"Prediction numpy: {pred}, prediction torch: {pred2.numpy()}" )
