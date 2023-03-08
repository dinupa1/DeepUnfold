#
# Iterative Bayesian unfolding with DNN
# dinupa3@gmail.com
#

import numpy as np
import matplotlib.pyplot as plt

import uproot
import awkward as ak
import hist
from hist import Hist

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from numba import njit

from MultiFold import Net, MultiFold

# open simulation events
theta0 = uproot.open("data.root:theta0")
theta0_G = theta0.arrays(["true_mass", "true_pT", "true_xF", "true_phi", "true_costh"], library="pd").to_numpy()
theta0_S = theta0.arrays(["mass", "pT", "xF", "phi", "costh"], library="pd").to_numpy()
weight_MC = theta0["weight"].array(library="np")

# open data events
theta1 = uproot.open("data.root:theta1")
theta1_G = theta1.arrays(["true_mass", "true_pT", "true_xF", "true_phi", "true_costh"], library="pd").to_numpy()
theta1_S = theta1.arrays(["mass", "pT", "xF", "phi", "costh"], library="pd").to_numpy()
weight_Data = theta1["weight"].array(library="np")

# plot the initial data
fig = plt.figure(figsize=(12, 5))

fig, ax = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True, sharey=False)

bin_edge = np.linspace(3.0, 9.0, 21)

# detector level
ax[0].set_title("Detector Level")
ax[0].hist(theta0_S[theta0_S[:, 0] > 0.][:, 0], bins=bin_edge, weights=weight_MC[theta0_S[:, 0] > 0.], histtype="step", label="simulation")
ax[0].hist(theta1_S[theta1_S[:, 0] > 0.][:, 0], bins=bin_edge, weights=weight_Data[theta1_S[:, 0] > 0.], histtype="step", label="data")
ax[0].set_ylabel("counts [a. u.]")
ax[0].set_xlabel("mass [GeV/c^2]")
ax[0].legend()

# particle level
ax[1].set_title("Particle Level")
ax[1].hist(theta0_G[:, 0], bins=bin_edge, weights=weight_MC, histtype="step", label="true")
ax[1].hist(theta1_G[:, 0], bins=bin_edge, weights=weight_Data, histtype="step", label="unkown")
ax[1].set_ylabel("counts [a. u.]")
ax[1].set_xlabel("mass [GeV/c^2]")
ax[1].legend()

plt.savefig("imgs/initial_data.png")
plt.close("all")

# iterative bayesian unfolding
weight_push, weight_pull, loss = MultiFold(epochs=40, iteration=4, theta0_G=theta0_G,
                                           theta0_S=theta0_S, theta_unknown_S=theta1_S,
                                           weight_MC=weight_MC, weight_Data=weight_Data)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(loss["train_1"], label="detector-training")
ax.plot(loss["val_1"], label="particle-validating")
ax.plot(loss["train_2"], label="detector-training")
ax.plot(loss["val_2"], label="particle-validating")
ax.set_xlabel("epochs")
ax.set_ylabel("error")
ax.legend()
ax.set_title("step 1")
plt.savefig("imgs/error.png")
plt.close("all")

# plot the results: mass
fig = plt.figure(figsize=(12, 5))

fig, ax = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True, sharey=False)

bin_edge = np.linspace(3.0, 9.0, 21)

# detector level
ax[0].set_title("Detector Level")
ax[0].hist(theta0_S[theta0_S[:, 0] > 0.][:, 0], bins=bin_edge, weights=weight_MC[theta0_S[:, 0] > 0.], histtype="step", label="simulation")
ax[0].hist(theta1_S[theta1_S[:, 0] > 0.][:, 0], bins=bin_edge, weights=weight_Data[theta1_S[:, 0] > 0.], histtype="step", label="data")
ax[0].hist(theta0_S[theta0_S[:, 0] > 0.][:, 0], bins=bin_edge, weights=weight_pull[theta0_S[:, 0] > 0.], histtype="step", label="unfolded")
ax[0].set_ylabel("counts [a. u.]")
ax[0].set_xlabel("mass [GeV/c^2]")
ax[0].legend()

# particle level
ax[1].set_title("Particle Level")
ax[1].hist(theta0_G[:, 0], bins=bin_edge, weights=weight_MC, histtype="step", label="true")
ax[1].hist(theta1_G[:, 0], bins=bin_edge, weights=weight_Data, histtype="step", label="unkown")
ax[1].hist(theta0_G[:, 0], bins=bin_edge, weights=weight_push, histtype="step", label="unfolded")
ax[1].set_ylabel("counts [a. u.]")
ax[1].set_xlabel("mass [GeV/c^2]")
ax[1].legend()

plt.savefig("imgs/unfold_mass.png")
plt.close("all")


# plot the results: phi
fig = plt.figure(figsize=(12, 5))

fig, ax = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True, sharey=False)

bin_edge = np.linspace(-np.pi, np.pi, 21)

# detector level
ax[0].set_title("Detector Level")
ax[0].hist(theta0_S[theta0_S[:, 0] > 0.][:, 3], bins=bin_edge, weights=weight_MC[theta0_S[:, 0] > 0.], histtype="step", label="simulation")
ax[0].hist(theta1_S[theta1_S[:, 0] > 0.][:, 3], bins=bin_edge, weights=weight_Data[theta1_S[:, 0] > 0.], histtype="step", label="data")
ax[0].hist(theta0_S[theta0_S[:, 0] > 0.][:, 3], bins=bin_edge, weights=weight_pull[theta0_S[:, 0] > 0.], histtype="step", label="unfolded")
ax[0].set_ylabel("counts [a. u.]")
ax[0].set_xlabel("phi [rad]")
ax[0].legend()

# particle level
ax[1].set_title("Particle Level")
ax[1].hist(theta0_G[:, 3], bins=bin_edge, weights=weight_MC, histtype="step", label="true")
ax[1].hist(theta1_G[:, 3], bins=bin_edge, weights=weight_Data, histtype="step", label="unkown")
ax[1].hist(theta0_G[:, 3], bins=bin_edge, weights=weight_push, histtype="step", label="unfolded")
ax[1].set_ylabel("counts [a. u.]")
ax[1].set_xlabel("phi [rad]")
ax[1].legend()

plt.savefig("imgs/unfold_phi.png")
plt.close("all")


# plot the results: costh
fig = plt.figure(figsize=(12, 5))

fig, ax = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True, sharey=False)

bin_edge = np.linspace(-1.0, 1.0, 21)

# detector level
ax[0].set_title("Detector Level")
ax[0].hist(theta0_S[theta0_S[:, 0] > 0.][:, 4], bins=bin_edge, weights=weight_MC[theta0_S[:, 0] > 0.], histtype="step", label="simulation")
ax[0].hist(theta1_S[theta1_S[:, 0] > 0.][:, 4], bins=bin_edge, weights=weight_Data[theta1_S[:, 0] > 0.], histtype="step", label="data")
ax[0].hist(theta0_S[theta0_S[:, 0] > 0.][:, 4], bins=bin_edge, weights=weight_pull[theta0_S[:, 0] > 0.], histtype="step", label="unfolded")
ax[0].set_ylabel("counts [a. u.]")
ax[0].set_xlabel("phi [rad]")
ax[0].legend()

# particle level
ax[1].set_title("Particle Level")
ax[1].hist(theta0_G[:, 4], bins=bin_edge, weights=weight_MC, histtype="step", label="true")
ax[1].hist(theta1_G[:, 4], bins=bin_edge, weights=weight_Data, histtype="step", label="unkown")
ax[1].hist(theta0_G[:, 4], bins=bin_edge, weights=weight_push, histtype="step", label="unfolded")
ax[1].set_ylabel("counts [a. u.]")
ax[1].set_xlabel("phi [rad]")
ax[1].legend()

plt.savefig("imgs/unfold_costh.png")
plt.close("all")