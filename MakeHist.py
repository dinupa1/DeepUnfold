# dinupa3@gmail.com
# 31 Jan 2023
#

import numpy as np
import matplotlib.pyplot as plt

import uproot
import hist
from hist import Hist
import numba
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

    hist2D = Hist(hist.axis.Regular(20, -pi, pi, name="phi"),
                  hist.axis.Regular(20, -1., 1., name="costh"))

    lambda_array = np.random.normal(0.5, 0.01, nsamples)
    mu_array = np.random.normal(0.0, 0.01, nsamples)
    nu_array = np.random.normal(0.0, 0.01, nsamples)

    for i in range(nsamples):
        print("sample = {}, lambda = {:.3f}, mu = {:.3f}, nu = {:.3f}".format(i, lambda_array[i], mu_array[i], nu_array[i]))
        hist2D.fill(data["phi"], data["costh"], weight=data["weight"]* weight2D(lambda_array[i], mu_array[i], nu_array[i], data["true_phi"], data["true_costh"]))

        hist_name = "hist_{}_{}_{}".format(lambda_array[i], mu_array[i], nu_array[i])

        out_file[hist_name] = hist2D

    print("**** monte-carlo sampling done ****")


nsamples = 100

tree = uproot.open("DY_DUMP_4pi_GMC_Jan08_LD2.root:result_mc")

data = tree.arrays(["weight", "phi", "costh", "true_phi", "true_costh"],
                       "(fpga1==1) & (4.5 < true_mass) & (true_mass < 8.0) & (mass > 0.)", library="np")

print("****     *****")
print(data["weight"].shape)
print("****     ****")

out_file = uproot.recreate("dummy.root", compression=uproot.ZLIB(4))

sample_mc(nsamples, data, out_file)
