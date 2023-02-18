import numpy as np
import matplotlib.pyplot as plt

import uproot
import awkward as ak
import hist
from hist import Hist

import torch
import torch.nn as nn


# some globel const
pi = np.pi
bins = 20


train_tree = uproot.open("data/toy.root:train_tree")
train_true = uproot.open("data/toy.root:train_true")

val_tree = uproot.open("data/toy.root:val_tree")
val_true = uproot.open("data/toy.root:val_true")

test_tree = uproot.open("data/toy.root:test_tree")
test_true = uproot.open("data/toy.root:test_true")


def plot_hist(bins, bin_min, bin_max, x, w, plot_name, fmt):
    bin_value = x * w
    bin_count = np.sum(bin_value, axis=0)
    bin_error = np.sqrt(np.sum(bin_value * bin_value, axis=0))

    bin_edge = np.linspace(-pi, pi, bins + 1)
    bin_cent = np.array([(bin_edge[i] + bin_edge[i + 1]) / 2 for i in range(bins)])

    return plt.errorbar(bin_cent, bin_count, yerr=bin_error, fmt=fmt, label=plot_name)


plot_hist(bins, -pi, pi, train_true["y"].array(library="np"), train_true["w"].array(library="np"), "train_true", "s")
plot_hist(bins, -pi, pi, train_tree["y"].array(library="np"), train_tree["w"].array(library="np"), "y_train", "o")
Hist(hist.axis.Regular(bins, -pi, pi, name="phi [rad]")).fill(train_tree["x"].array(library="np").ravel(), weight=train_tree["w"].array(library="np").ravel()).plot(label="train_reco")
plt.xlabel("phi [rad]")
plt.ylabel("counts")
plt.legend()
plt.save("imgs/test.png")
# plt.show()


import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class Net(nn.Module):
    def __init__(self, input_dim: int=1, output_dim: int=bins, hidden_dim: int = 50):
        super(Net, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, output_dim, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.fc1(x)
        return x


net = Net()

weight_train = train_tree["w"].array(library="np").reshape(-1, 1)
weight_true = train_true["w"].array(library="np").reshape(-1, 1)


optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)
criterion = nn.CrossEntropyLoss()
batch_size = 50000


train_dataset = TensorDataset(torch.Tensor(train_tree["x"].array(library="np")), torch.Tensor(train_tree["y"].array(library="np")))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

def weight_class(xarray, weight):
    p = np.sum(xarray* weight, axis=0)
    return p/np.sum(p)


iterations = 20
epochs = 200

loss = []

for i in range(iterations):
    # print("******* iteration : {} *********".format(i))
    for epoch in range(epochs):
        net.train()
        for inputs, targets in train_dataloader:
            optimizer.zero_grad()

            outputs = net(inputs)
            error = criterion(outputs, targets)
            error.backward()
            optimizer.step()

        loss.append(error.detach().numpy())
        # print("epoch : {} & error : {}".format(epoch, error))

    net.eval()
    outputs = net(torch.Tensor(train_tree["x"].array(library="np"))).detach().numpy()
    weight0 = weight_class(train_true["y"].array(library="np"), weight_true)
    weight1 = weight_class(train_tree["y"].array(library="np"), weight_train)
    criterion.weight = torch.Tensor(weight0 / weight1)


plt.plot(loss)
plt.xlabel("ephoch and iteration")
plt.ylabel("counts")
plt.savefig("imgs/loss.png")
# plt.show()


X_test = test_tree["x"].array(library="np")
y_test = test_tree["y"].array(library="np")
weight_test = test_tree["w"].array(library="np")

y_true = test_true["y"].array(library="np")
weight_true = test_true["w"].array(library="np")

net.eval()

outputs = net(torch.Tensor(X_test)).detach().numpy()

plot_hist(bins, -pi, pi, y_test, weight_test, "test reco", "s")
plot_hist(bins, -pi, pi, outputs, weight_test, "Bayes", "o")
plot_hist(bins, -pi, pi, y_true, weight_true, "test true", "*")
plt.legend()
plt.savefig("imgs/prediction.png")
# plt.show()