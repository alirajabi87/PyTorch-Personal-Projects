import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

import time

df = pd.read_csv('../../PYTORCH_NOTEBOOKS/Data/NYCTaxiFares.csv')


# print(df.head())


def harversine(phi1, phi2, lambda1, lambda2):
    """
    The haversine function computes the distance between two latitude and longitude
    :param phi1 and phi2: φ1, φ2 are the latitude of point 1 and latitude of point 2 (in radians),
    :param lambda1 and lambda2: λ1, λ2 are the longitude of point 1 and longitude of point 2 (in radians).
    :return: distance
    """
    phi1 = np.radians(phi1)
    phi2 = np.radians(phi2)
    lambda1 = np.radians(lambda1)
    lambda2 = np.radians(lambda2)

    R_earth = 6371  # km
    hav = lambda x: (np.sin(x / 2)) ** 2
    d = 2 * R_earth * np.arcsin(np.sqrt(hav(phi2 - phi1) + np.cos(phi1) * np.cos(phi2) * hav(lambda2 - lambda1)))
    return d


df['dist_km'] = harversine(df['pickup_latitude'], df['dropoff_latitude'], df['pickup_longitude'],
                           df['dropoff_longitude'])
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['EDTdate'] = df['pickup_datetime'] - pd.Timedelta(hours=4)
df['Hour'] = df['EDTdate'].dt.hour
df['AMorPM'] = np.where(df['Hour'] > 12, 'pm', 'am')
df['Weekday'] = df['EDTdate'].dt.dayofweek  # dt.strftime("%a") #
df['AMorPM'] = df['AMorPM'].map({'am': 0, 'pm': 1})
cat_col = 'Hour  AMorPM  Weekday'.split()
for cat in cat_col:
    df[cat] = df[cat].astype('category')

df = df.drop('pickup_datetime pickup_longitude  pickup_latitude  dropoff_longitude  dropoff_latitude EDTdate'.split(),
             axis=1)
# print(df.dtypes)
# print(df['Weekday'].head())

y_label = df['fare_amount']

hr = df['Hour'].cat.codes.values
ampm = df['AMorPM'].cat.codes.values
wkdy = df['Weekday'].cat.codes.values

cats = np.stack([df[col].cat.codes.values for col in cat_col], axis=1)
cats = torch.tensor(cats, dtype=torch.int64)
cont_col = 'passenger_count dist_km'.split()

conts = np.stack([df[col].values for col in cont_col], axis=1)
conts = torch.tensor(conts, dtype=torch.float)

y = torch.tensor(y_label.values, dtype=torch.float).reshape(-1, 1)
# print(cats.shape, conts.shape, y.shape)

cat_sizes = [len(df['Hour'].cat.categories), len(df['AMorPM'].cat.categories), len(df['Weekday'].cat.categories)]
emb_size = [(size, min(50, (size + 1) // 2)) for size in cat_sizes]


class Model(nn.Module):
    def __init__(self, embedding_sizes, n_continue, output_size, hidden_layers, DropOut=0.4):
        # layers = [200, 100, 50 ] number of hidden neurons
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embedding_sizes])
        self.emb_drop = nn.Dropout(DropOut)
        self.bn_cont = nn.BatchNorm1d(n_continue)

        layerList = []
        n_embs = sum([nf for ni, nf in embedding_sizes])
        n_in = n_embs + n_continue

        for i in hidden_layers:
            layerList.append(nn.Linear(n_in, i))
            layerList.append(nn.ReLU(inplace=True))
            layerList.append(nn.BatchNorm1d(i))
            layerList.append(nn.Dropout(DropOut))
            n_in = i
        layerList.append(nn.Linear(hidden_layers[-1], output_size))
        self.layers = nn.Sequential(*layerList)

    def forward(self, x_cat, x_cont):
        embeddings = []

        for i, e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:, i]))

        x_cat = torch.cat(embeddings, 1)
        x_cat = self.emb_drop(x_cat)
        x_cont = self.bn_cont(x_cont)

        x = torch.cat([x_cat, x_cont], 1)
        x = self.layers(x)
        return x


def main():
    torch.manual_seed(33)
    model = Model(emb_size, conts.shape[1], output_size=1, hidden_layers=[200, 100], DropOut=0.4)
    critirion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    batch_size = 120000
    test_size = int(batch_size * 0.2)

    # train test split
    cat_train = cats[:batch_size - test_size]
    cat_test = cats[batch_size - test_size:batch_size]
    con_train = conts[:batch_size - test_size]
    con_test = conts[batch_size - test_size:batch_size]
    y_train = y[:batch_size - test_size]
    y_test = y[batch_size - test_size:batch_size]

    startTime = time.time()
    EPOCHS = 600
    losses = []

    # for i in range(EPOCHS):
    #     i += 1
    #     y_pred = model(cat_train, con_train)
    #     loss = torch.sqrt(critirion(y_pred, y_train))
    #     losses.append(loss)
    #
    #     if not i%10:
    #         print(f"epoch: {i}, loss: {loss}")
    #
    #
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #
    # print(f"Duration: {(time.time()-startTime)/60}")
    # plt.plot(losses)
    # plt.show()
    # torch.save(model.state_dict(), 'Taxi.pt') # to save entire model just put model

    model.load_state_dict(torch.load('Taxi.pt'))
    ValidateModel(model, critirion, cat_test, con_test, y_test)

def ValidateModel(model, criterion, cat_test, con_test,  y_test):
    with torch.no_grad():
        y_val = model.forward(cat_test, con_test)
        loss = torch.sqrt(criterion(y_val, y_test))
        print(loss)
if __name__ == '__main__':
    main()

