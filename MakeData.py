#
# make simple data file for neural network training
# dinupa3@gmail.com
# 04March2023
#

import uproot
import awkward as ak
import numpy as np
from numba import njit

from sklearn.model_selection import train_test_split

@njit
def weight2D(Lambda, Mu, Nu, phi, costh):
    weight = 1+ Lambda* costh* costh+ Mu* 2* costh* np.sqrt(1- costh* costh)* np.cos(phi)+ (Nu/2)* (1 -costh* costh)* np.cos(2* phi)
    return weight


tree = uproot.open("data/DY_DUMP_4pi_GMC_Jan08_LD2.root:result_mc")

events = tree.arrays(["weight", "true_mass", "true_pT", "true_xF", "true_phi", "true_costh", "mass", "pT", "xF", "phi", "costh"],
                     "(fpga1==1) & (5.0 < true_mass) & (true_mass < 8.0)", library="pd").to_numpy()

theta0, theta1 = train_test_split(events, test_size=0.5, shuffle=True)

theta0 = np.nan_to_num(theta0, nan=-999.)
theta1 = np.nan_to_num(theta1, nan=-999.)

theta0 = ak.zip({
    "weight": theta0[:, 0],
    "true_mass": theta0[:, 1],
    "true_pT": theta0[:, 2],
    "true_xF": theta0[:, 3],
    "true_phi": theta0[:, 4],
    "true_costh": theta0[:, 5],
    "mass": theta0[:, 6],
    "pT": theta0[:, 7],
    "xF": theta0[:, 8],
    "phi": theta0[:, 9],
    "costh": theta0[:, 10],
})

theta1 = ak.zip({
    # for now lets set lambda = 0.5, mu = 0.2, nu = -0.3
    "weight": theta1[:, 0]* weight2D(0.5, 0.2, -0.3, theta1[:, 4], theta1[:, 5]),
    "true_mass": theta1[:, 1],
    "true_pT": theta1[:, 2],
    "true_xF": theta1[:, 3],
    "true_phi": theta1[:, 4],
    "true_costh": theta1[:, 5],
    "mass": theta1[:, 6],
    "pT": theta1[:, 7],
    "xF": theta1[:, 8],
    "phi": theta1[:, 9],
    "costh": theta1[:, 10],
})

theta0 = theta0[:1000000]
theta1 = theta1[(5.3 < theta1.true_mass) & (theta1.true_mass < 7.7)][:1000000]

outputs = uproot.recreate("data.root", compression=uproot.ZLIB(4))
outputs["theta0"] = theta0
outputs["theta1"] = theta1