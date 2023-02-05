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

    # lambda_array = np.random.uniform(-1., 1., nsamples)
    # mu_array = np.random.uniform(-1., 1., nsamples)
    # nu_array = np.random.uniform(-1., 1., nsamples)

    lambda_array = np.random.normal(0.2, 0.01, nsamples)
    mu_array = np.random.normal(0.1, 0.01, nsamples)
    nu_array = np.random.normal(-0.1, 0.01, nsamples)

    hist_array = []

    for i in range(nsamples):

        hist2D = Hist(hist.axis.Regular(20, -pi, pi, name="phi"),
                  hist.axis.Regular(20, -1., 1., name="costh"))

        print("sample = {}, lambda = {:.3f}, mu = {:.3f}, nu = {:.3f}".format(i, lambda_array[i], mu_array[i], nu_array[i]))

        hist2D.fill(data["phi"], data["costh"], weight=data["weight"]* weight2D(lambda_array[i], mu_array[i], nu_array[i], data["true_phi"], data["true_costh"]))

        hist_name = "hist_{}_{}_{}".format(lambda_array[i], mu_array[i], nu_array[i])

        out_file[hist_name] = hist2D

    print("**** monte-carlo sampling done ****")


nsamples = 50

tree = uproot.open("data/DY_DUMP_4pi_GMC_Jan08_LD2.root:result_mc")

data = tree.arrays(["weight", "phi", "costh", "true_phi", "true_costh"],
                       "(fpga1==1) & (4.5 < true_mass) & (true_mass < 8.0) & (mass > 0.)", library="np")


out_file = uproot.recreate("data/dummy.root", compression=uproot.ZLIB(4))

sample_mc(nsamples, data, out_file)
