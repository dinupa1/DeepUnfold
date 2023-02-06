# dinupa3@gmail.com
# 31 Jan 2023
#

import numpy as np
import matplotlib.pyplot as plt

import uproot
import hist
from hist import Hist
import numba
import awkward as ak
from numba import njit

pi = np.pi

# define a function to appy weight -> go parallel
@njit(parallel=True)
def weight2D(lam, mu, nu, phi, costh):

    result = 1+ lam* costh* costh+ mu* 2* costh* np.sqrt(1- costh*costh)* np.cos(phi)+ nu* (1 - costh* costh)* np.cos(2* phi)

    return result


# define a funcion for monte-carlo sampling
# @njit()
def sample_mc(nsamples, data, out_file):

    # _lambda_ = np.random.uniform(-1., 1., nsamples)
    # _mu_ = np.random.uniform(-1., 1., nsamples)
    # _nu_ = np.random.uniform(-1., 1., nsamples)

    _lambda_ = np.random.normal(0.2, 0.01, nsamples)
    _mu_ = np.random.normal(-0.1, 0.01, nsamples)
    _nu_ = np.random.normal(0.1, 0.01, nsamples)

    for i in range(nsamples):

        hist2D = Hist(hist.axis.Regular(20, -pi, pi, name="phi"),
                  hist.axis.Regular(20, -1., 1., name="costh"))

        print("sample = {}, lambda = {:.3f}, mu = {:.3f}, nu = {:.3f}".format(i, _lambda_[i], _mu_[i], _nu_[i]))

        hist2D.fill(data["phi"], data["costh"], weight=data["weight"]* weight2D(_lambda_[i], _mu_[i], _nu_[i], data["true_phi"], data["true_costh"]))

        hist_name = "hist_{}".format(i)

        out_file[hist_name] = hist2D

        hist2D.reset()


    out_file["tree"] = ak.zip({
        "lambda": _lambda_,
        "mu": _mu_,
        "nu": _nu_,
        })

    print("**** monte-carlo sampling done ****")


nsamples = 30000

tree = uproot.open("data/DY_DUMP_4pi_GMC_Jan08_LD2.root:result_mc")

data = tree.arrays(["weight", "phi", "costh", "true_phi", "true_costh"],
                       "(fpga1==1) & (4.5 < true_mass) & (true_mass < 8.0) & (mass > 0.)", library="np")


out_file = uproot.recreate("data/test_hist_2_.root", compression=uproot.ZLIB(4))

sample_mc(nsamples, data, out_file)
