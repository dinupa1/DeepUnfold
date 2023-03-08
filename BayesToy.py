import numpy as np
import matplotlib.pyplot as plt

import uproot
import awkward as ak
import hist
from hist import Hist

import torch
import torch.nn as nn

import Utils
from Utils import BayesData

# some global const
pi = np.pi
phiBins = 10
costhBins = 10
bins = phiBins* costhBins


# tree = uproot.open("data/toy.root:simple")
# simple = tree.arrays(["weight", "mass", "pT", "xF", "phi", "costh", "true_phi"], library="pd").to_numpy()
#
# data = BayesData(simple)

train_tree = uproot.open("data/bayesData.root:train_tree")
test_tree = uproot.open("data/bayesData.root:test_tree")

X_train = train_tree.arrays(["mass", "pT", "xF", "phi", "costh"], library="pd").to_numpy()
weight_train = train_tree.arrays(["weight"], library="pd").to_numpy()
y_train = train_tree["y"].array(library="pd").to_numpy(dtype=np.float32)

X_test = test_tree.arrays(["mass", "pT", "xF", "phi", "costh"], library="pd").to_numpy()
weight_test = test_tree.arrays(["weight"], library="pd").to_numpy()
y_test = test_tree["y"].array(library="pd").to_numpy(dtype=np.float32)

print("train shapes : x = {} & weight = {} & y = {}".format(X_train.shape, weight_train.shape, y_train.shape))
print("train shapes : x = {} & weight = {} & y = {}".format(X_test.shape, weight_test.shape, y_test.shape))
print(np.min(X_train), np.max(X_train), np.min(y_train), np.max(y_train))
print(np.min(X_test), np.max(X_test), np.min(y_test), np.max(y_test))

def plot_hist(bins, x, w, plot_name, fmt):
    bin_value = x * w
    bin_count = np.sum(bin_value, axis=0)
    bin_error = np.sqrt(np.sum(x* w* w, axis=0))
    # bin_count = bin_count* (1/effi)

    bin_class = np.linspace(1., 100., bins)

    return plt.errorbar(bin_class, bin_count, yerr=bin_error, fmt=fmt, label=plot_name)


# plot_hist(bins, -pi, pi, data.true_train, data.true_train_weight, "train_true", "s")
# plot_hist(bins, -pi, pi, data.y_train, data.weight_train, "y_train", "o")
# Hist(hist.axis.Regular(bins, -pi, pi, name="phi [rad]")).fill(data.X_train[:, 3].ravel(), weight=data.weight_train.ravel()).plot(label="train_reco")
# plt.xlabel("phi [rad]")
# plt.ylabel("counts")
# plt.legend()
# plt.savefig("imgs/test.png")
# # plt.show()
# plt.close("all")

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class Net(nn.Module):
    def __init__(self, input_dim: int=5, output_dim: int=bins, hidden_dim=500):
        super(Net, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=0.4),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=0.4),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=0.4),
            nn.Linear(hidden_dim, output_dim, bias=True),
            nn.Softmax(dim=1)
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.fc1(x)
        return x



net = Net()

total_trainable_params = sum(p.numel() for p in net.parameters())
print('total trainable params:', total_trainable_params)


optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
# criterion = nn.CrossEntropyLoss()
criterion = nn.NLLLoss()
# criterion = nn.MultiLabelSoftMarginLoss()
batch_size = 1000


train_dataset = TensorDataset(torch.Tensor(X_train[X_train[:, 0] > 0.]), torch.Tensor(y_train[X_train[:, 0] > 0.]))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

def weight_class(xarray, weight):
    return np.sum(xarray* weight, axis=0)
    # return np.sum(xarray, axis=0)


iterations = 4
epochs = 500

# loss = []

# class_weight = np.sum(data.y_train, axis=0)/np.sum(data.true_train, axis=0)
# criterion.weight = torch.Tensor(class_weight)
# print(class_weight)

# print(weight_class(data.y_train, data.weight_train)/weight_class(data.true_train, data.true_train_weight))

# initial training
# for epoch in range(epochs):
#     net.train()
#     for inputs, targets in train_dataloader:
#         optimizer.zero_grad()
#         outputs = net(inputs)
#
#         # print(outputs)
#
#         # error = criterion(outputs, targets)
#         error = criterion(torch.log(outputs), torch.argmax(targets, dim=1))
#         error.backward()
#         optimizer.step()
#     # loss.append(error.detach().numpy())
#     # print("epoch : {} & error : {}".format(epoch, error))


loss = []

for i in range(iterations):
    print("******* iteration : {} *********".format(i))
    for epoch in range(epochs):
        net.train()
        for inputs, targets in train_dataloader:
            optimizer.zero_grad()

            outputs = net(inputs)
            error = criterion(torch.log(outputs), torch.argmax(targets, dim=1))
            # error = criterion(outputs, torch.argmax(targets, dim=-1))
            error.backward()
            optimizer.step()

        loss.append(error.detach().numpy())
        # print(error)
        # print("epoch : {} & error : {}".format(epoch, error))

    net.eval()
    outputs = net(torch.Tensor(X_train[X_train[:, 0] > 0.])).detach().numpy()
    weight0 = weight_class(y_train, weight_train)
    weight1 = weight_class(outputs, weight_train[X_train[:, 0] > 0.])
    update_weight = weight1 / weight0
    # print("effi update : {}".format(effi_update))
    # print("update weight: {}".format(update_weight))
    criterion.weight = torch.Tensor(update_weight)
    # print("updated weight : {}".format(criterion.weight))
    # sample_weight = torch.Tensor(weight1 / weight0)


plt.plot(loss)
plt.xlabel("ephoch and iteration")
plt.ylabel("counts")
plt.savefig("imgs/loss.png")
# plt.show()
plt.close("all")


net.eval()
outputs = net(torch.Tensor(X_test[X_test[:, 0] > 0.])).detach().numpy()

plot_hist(bins,y_test[X_test[:, 0] > 0.], weight_test[X_test[:, 0] > 0.],"test reco", "s")
plot_hist(bins,outputs, weight_test[X_test[:, 0] > 0.]* (1/criterion.weight.detach().numpy()), "Bayes", "o")
plot_hist(bins, y_test, weight_test, "test true", "*")
plt.legend()
plt.savefig("imgs/prediction.png")
# plt.show()
plt.close("all")

bin_edge = np.linspace(-pi, pi, bins + 1)
bin_cent = np.array([(bin_edge[i] + bin_edge[i + 1]) / 2 for i in range(bins)])

test_bin_reco = np.sum(y_test[X_test[:, 0] > 0.]* weight_test[X_test[:, 0] > 0.], axis=0)
test_bin_true = np.sum(y_test* weight_test, axis=0)
test_bayes = np.sum(outputs* weight_test[X_test[:, 0] > 0.]* (1/criterion.weight.detach().numpy()), axis=0)
reco_bin_error = np.sqrt(np.sum(y_test[X_test[:, 0] > 0.]* weight_test[X_test[:, 0] > 0.]* weight_test[X_test[:, 0] > 0.], axis=0))
true_bin_error = np.sqrt(np.sum(y_test* weight_test* weight_test, axis=0))
bayes_bin_error = np.sqrt(np.sum(outputs* weight_test[X_test[:, 0] > 0.]*(1/criterion.weight.detach().numpy())* weight_test[X_test[:, 0] > 0.]*(1/criterion.weight.detach().numpy()), axis=0))

import ROOT
from ROOT import TH2D

test_reco_hist = TH2D("test_reco_hist", "; phi [rad]; costh", phiBins, -pi, pi, costhBins, -1., 1.)
test_true_hist = TH2D("test_true_hist", "; phi [rad]; costh", phiBins, -pi, pi, costhBins, -1., 1.)
test_bayes_hist = TH2D("test_bayes_hist", "; phi [rad]; costh", phiBins, -pi, pi, costhBins, -1., 1.)

for i in range(phiBins):
    for j in range(costhBins):
        test_true_hist.SetBinContent(i* costhBins + j + 1, test_bin_true[i* costhBins + j])
        test_true_hist.SetBinError(i* costhBins + j+ 1, true_bin_error[i* costhBins + j])
        test_reco_hist.SetBinContent(i* costhBins + j+ 1, test_bin_reco[i* costhBins + j])
        test_reco_hist.SetBinError(i* costhBins + j+ 1, reco_bin_error[i* costhBins + j])
        test_bayes_hist.SetBinContent(i* costhBins + j+ 1, test_bayes[i* costhBins + j])
        test_bayes_hist.SetBinError(i* costhBins + j+ 1, bayes_bin_error[i* costhBins + j])


outfile = uproot.recreate("data/result.root", compression=uproot.ZLIB(4))
outfile["train_tree"] = train_tree.arrays(["weight", "mass", "pT", "xF", "phi", "costh", "true_phi", "true_costh"], library="pd")
outfile["test_tree"] = test_tree.arrays(["weight", "mass", "pT", "xF", "phi", "costh", "true_phi", "true_costh"], library="pd")
outfile["test_reco_hist"] = test_reco_hist
outfile["test_true_hist"] = test_true_hist
outfile["test_bayes_hist"] = test_bayes_hist

# ./runGMC_root.py --grid --server=e906-db3.fnal.gov --port=3306 --raw-name=test --n-events=10000 --n-subruns=2 --geometry=geometry_G18_run3 --gmc-args="/set/beamYOffset 1.6 cm" --Target=H --EventPosition=Dump --Generator=DY --Acceptance=Acc --gmc-args="/set/fmagMultiplier -1.044" --gmc-args="/set/kmagMultiplier -1.025" --grid-args="--expected-lifetime=6h" --gmc-args="/set/recordTo root" --outdir=/pnfs/e906/scratch/users/dinupa/test --first-subrun=2 --osg

# jobsub_submit: error: argument executable: invalid verify_executable_starts_with_file_colon value: '--subgroup production'