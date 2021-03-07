import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F

df = pd.read_csv('../../PYTORCH_NOTEBOOKS/Data/income.csv')
allColumns = ['age', 'sex', 'education', 'education-num', 'marital-status',
              'workclass', 'occupation', 'hours-per-week', 'income', 'label']

# Shuffle the dataset
df = shuffle(df, random_state=101)
df.reset_index(drop=True, inplace=True)

# Data engineering
def age(x):
    if x <= 20:
        return 0
    elif x > 20 and x <= 30:
        return 1
    elif x > 30 and x <= 40:
        return 2
    elif x > 40 and x <= 50:
        return 3
    elif x > 50 and x <= 60:
        return 4
    elif x > 60 and x <= 70:
        return 5
    elif x > 70 and x <= 80:
        return 6
    else:
        return 7


df['age'] = df['age'].apply(age)
df.drop(['education', 'income'], axis=1, inplace=True)
print(df.head())
print("\n\n")

# print(df.isnull().sum())

cat_col = "age sex education-num marital-status workclass occupation".split()
cont_col = ['hours-per-week']
y = torch.tensor(df['label'].values).flatten()

for col in cat_col:
    df[col] = df[col].astype('category')

# print(df.dtypes)

cats = np.stack([df[col].cat.codes.values for col in cat_col], axis=1)
cats = torch.tensor(cats, dtype=torch.int64).cuda()
print(cats.shape)

cats_size = [len(df[col].cat.categories) for col in cat_col]
emb_size = [(size, min(50, (size+1) // 2))for size in cats_size]
print(cats_size)
print(emb_size)


conts = np.stack([df[col].values for col in cont_col], axis=1)
conts = torch.tensor(conts, dtype=torch.float).cuda()
print(conts.shape)


# print(torch.cuda.memory_allocated())

class Model(nn.Module):
    def __init__(self, emb_size, n_continues, hiddenLayers, output, DropOut=0.4):
        super(Model, self).__init__()

        self.embed = nn.ModuleList([nn.Embedding(ni, nf) for ni,nf in emb_size])
        self.embed_drop = nn.Dropout(DropOut, inplace=True)

        self.BN_cont = nn.BatchNorm1d(n_continues)

        layerList = []
        n_emb = sum((nf for _,nf in emb_size))
        i_inp = n_emb+n_continues
        for i in hiddenLayers:
            layerList.append(nn.Linear(i_inp, i))
            layerList.append(nn.ReLU(inplace=True))
            layerList.append(nn.BatchNorm1d(i))
            layerList.append(nn.Dropout(DropOut, inplace=True))
            i_inp = i
        layerList.append(nn.Linear(hiddenLayers[-1], output))
        self.layers = nn.Sequential(*layerList)


    def forward(self, x_cat, x_cont):
        embeddings = []

        for i, e in enumerate(self.embed):
            embeddings.append(e(x_cat[:,i]))

        x_cat = torch.cat(embeddings, 1)
        x_cat = self.embed_drop(x_cat)
        x_cont = self.BN_cont(x_cont)

        X_input = torch.cat([x_cat, x_cont], 1).cuda()
        x = self.layers(X_input)
        return x

def main():
    torch.manual_seed(33)
    model = Model(emb_size, conts.shape[1], [256, 128, 64], 2, 0.4).cuda()
    print(model.eval())

    batch_size = int(1 * df.shape[0])
    test_size = int(0.2*batch_size)
    print(batch_size, test_size)
    # print(df.shape)
    cat_train = cats[:batch_size-test_size]
    cat_test = cats[batch_size-test_size:batch_size]
    con_train = conts[:batch_size-test_size]
    con_test = conts[batch_size - test_size:batch_size]
    y_train = y[:batch_size-test_size]

    y_test = y[batch_size - test_size:batch_size]


    y_train = torch.LongTensor(y_train).cuda()
    y_test = torch.LongTensor(np.array(y_test))

    import time
    startTime = time.time()
    EPOCHS = 300
    losses = []
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    for i in range(EPOCHS):
        i += 1
        y_pred = model(cat_train, con_train)
        loss = criterion(y_pred, y_train)
        losses.append(loss)

        if not i%10:
            print(f"epoch: {i:3}, loss: {loss.item():10.8f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Duration: {(time.time() - startTime) / 60}")
    torch.save(model.state_dict(), 'income.pt')
    plt.plot(losses)
    plt.show()

    with torch.no_grad():
        pred = model(cat_test, con_test).cpu()
        loss = criterion(pred, y_test)
        pred = np.argmax(pred.numpy(), axis=-1)
        print(classification_report(y_true=y_test.numpy(), y_pred=pred))

if __name__ == '__main__':
    main()