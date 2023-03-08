#
# custom data set
# dinupa3@gmail.com
# 23 Feb 2023
#

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from numba import njit

@njit
def reweight(m):
    weights = m/(1. - m)
    return weights.ravel()

class Net(nn.Module):
    def __init__(self, input_dim: int = 5, output_dim: int = 1, hidden_dim: int = 50):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        # x = nn.Dropout(p=0.4)(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        # x = nn.Dropout(p=0.4)(x)
        # x = self.fc3(x)
        # x = nn.ReLU()(x)
        x = self.fc4(x)
        x = nn.Sigmoid()(x)
        return x

def MultiFold(epochs: int = 20, iteration: int = 4, theta0_G = None, theta0_S = None, theta_unknown_S = None,
              weight_MC = None, weight_Data = None):

    batch_size = 5000

    theta0 = np.stack([theta0_G, theta0_S], axis=1)
    label0 = np.zeros(len(theta0))
    theta_unknown = np.stack([theta_unknown_S, theta_unknown_S], axis=1)
    label1 = np.ones(len(theta0_G))
    label_unknown = np.ones(len(theta_unknown_S))

    xvals_1 = np.concatenate((theta0_S, theta_unknown_S))
    yvals_1 = np.concatenate((label0, label_unknown)).reshape(-1, 1)
    xvals_2 = np.concatenate((theta0_G, theta0_G))
    yvals_2 = np.concatenate((label0, label1)).reshape(-1, 1)

    weight_pull = np.array([m for m in weight_MC])
    weight_push = np.array([m for m in weight_MC])
    weight_MC0 = np.array([m for m in weight_MC])

    loss ={}
    loss["train_1"], loss["val_1"], loss["train_2"], loss["val_2"] = [], [], [], []


    for i in range(iteration):
        print("iteration : {}".format(i+1))
        print("step 1 ...")

        model_1 = Net(hidden_dim=100)

        total_trainable_params = sum(p.numel() for p in model_1.parameters())
        print('total trainable params in model 1 : ', total_trainable_params)

        optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=0.001)

        weight_1 = np.concatenate((weight_push, weight_Data)).reshape(-1, 1)

        X_train_1, X_test_1, Y_train_1, Y_test_1, w_train_1, w_test_1 = train_test_split(xvals_1, yvals_1,
                                                                                        weight_1, test_size=0.3, shuffle=True)

        train_dataset_1 = TensorDataset(torch.Tensor(X_train_1[X_train_1[:, 0] > 0.]),
                                        torch.Tensor(Y_train_1[X_train_1[:, 0] > 0.]),
                                        torch.Tensor(w_train_1[X_train_1[:, 0] > 0.]))
        train_dataloader_1 = DataLoader(train_dataset_1, batch_size=batch_size, shuffle=True)

        test_dataset_1 = TensorDataset(torch.Tensor(X_test_1[X_test_1[:, 0] > 0.]),
                                        torch.Tensor(Y_test_1[X_test_1[:, 0] > 0.]),
                                        torch.Tensor(w_test_1[X_test_1[:, 0] > 0.]))
        test_dataloader_1 = DataLoader(test_dataset_1, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            model_1.train()
            errD, n = 0., 0.
            for inputs, targets, weights in train_dataloader_1:
                optimizer_1.zero_grad()
                criterion = nn.BCELoss(weight=weights)
                outputs = model_1(inputs)
                error = criterion(outputs, targets)
                error.backward()
                optimizer_1.step()
                errD += error.item()
                n += 1.
            loss["train_1"].append(errD/n)
            # print("epoch : {} & error : {}".format(epoch, errD))

            model_1.eval()
            errV, m = 0., 0.
            for inputs, targets, weights in test_dataloader_1:
                criterion = nn.BCELoss(weight=weights)
                outputs = model_1(inputs)
                error = criterion(outputs, targets)
                errV += error.item()
                m += 1.
            loss["val_1"].append(errV/m)

        print("update weight pull ...")
        model_1.eval()
        p = model_1(torch.Tensor(theta0_S)).detach().numpy()
        weight_pull = weight_push * reweight(p)
        # weight_pull[theta0_S[:, 0] < 0.] = 1.0
        weight_pull[theta0_S[:, 0] < 0.] = weight_MC0[theta0_S[:, 0] < 0.]

        print("step 2 ...")

        model_2 = Net(hidden_dim=100)

        total_trainable_params = sum(p.numel() for p in model_2.parameters())
        print('total trainable params in model 2 : ', total_trainable_params)

        optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=0.001)

        weight_2 = np.concatenate((weight_MC, weight_pull)).reshape(-1, 1)

        X_train_2, X_test_2, Y_train_2, Y_test_2, w_train_2, w_test_2 = train_test_split(xvals_2, yvals_2,
                                                                                         weight_2, test_size=0.3,
                                                                                         shuffle=True)

        train_dataset_2 = TensorDataset(torch.Tensor(X_train_2),
                                        torch.Tensor(Y_train_2),
                                        torch.Tensor(w_train_2))
        train_dataloader_2 = DataLoader(train_dataset_2, batch_size=batch_size, shuffle=True)

        test_dataset_2 = TensorDataset(torch.Tensor(X_test_2),
                                       torch.Tensor(Y_test_2),
                                       torch.Tensor(w_test_2))
        test_dataloader_2 = DataLoader(test_dataset_2, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            model_2.train()
            errD, n = 0., 0.
            for inputs, targets, weights in train_dataloader_2:
                optimizer_2.zero_grad()
                criterion = nn.BCELoss(weight=weights)
                outputs = model_2(inputs)
                error = criterion(outputs, targets)
                error.backward()
                optimizer_2.step()
                errD += error.item()
                n += 1.
            loss["train_2"].append(errD/n)
            # print("epoch : {} & error : {}".format(epoch, errD))

            model_2.eval()
            errV, m = 0., 0.
            for inputs, targets, weights in test_dataloader_2:
                criterion = nn.BCELoss(weight=weights)
                outputs = model_2(inputs)
                error = criterion(outputs, targets)
                errV += error.item()
                m += 1.
            loss["val_2"].append(errV/m)

        print("update weight push ...")
        model_2.eval()
        p = model_2(torch.Tensor(theta0_G)).detach().numpy()
        # weight_push = weight_MC0 * reweight(p)
        weight_push = weight_pull * reweight(p)


    return weight_push, weight_pull, loss