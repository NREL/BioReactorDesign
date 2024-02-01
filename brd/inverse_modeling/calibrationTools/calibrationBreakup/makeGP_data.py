import os
import pickle
import sys

import numpy as np

sys.path.append("util")
import matplotlib.pyplot as plt
from plotsUtil import *
from SALib.analyze import delta

root = "."


def limiter(rec):
    return False
    # if rec['objective']>0.2:
    #   return True
    # else:
    #   return False


with open(os.path.join(root, "error.pkl"), "rb") as fp:
    error = pickle.load(fp)


var_names = ["surfaceTension", "henry", "coal_eff", "breakup_eff"]
nParams = len(var_names)

nSim = 0
for indSim in error:
    err = error[indSim]
    if limiter(err):
        continue
    nSim += 1
sum_mse_gh_17 = np.zeros(nSim)
sum_mse_gh_19 = np.zeros(nSim)
sum_mse_xco2_17 = np.zeros(nSim)
sum_mse_xco2_19 = np.zeros(nSim)
params = np.zeros((nSim, nParams))


iSim = 0
for indSim in error:
    err = error[indSim]
    if limiter(err):
        continue
    sum_mse_gh_17[iSim] = np.sum((err["err17_gh"]) ** 2)
    sum_mse_gh_19[iSim] = np.sum((err["err19_gh"]) ** 2)
    sum_mse_xco2_17[iSim] = np.sum((err["err17_xco2"]) ** 2)
    sum_mse_xco2_19[iSim] = np.sum((err["err19_xco2"]) ** 2)
    for iname, name in enumerate(var_names):
        params[iSim, iname] = err[name]
    iSim += 1


np.savez(
    "train_data_gh_17.npz",
    x=params,
    y=sum_mse_gh_17,
    nobs=len(error[0]["err17_gh"]),
)
np.savez(
    "train_data_gh_19.npz",
    x=params,
    y=sum_mse_gh_19,
    nobs=len(error[0]["err19_gh"]),
)
np.savez(
    "train_data_xco2_17.npz",
    x=params,
    y=sum_mse_xco2_17,
    nobs=len(error[0]["err17_xco2"]),
)
np.savez(
    "train_data_xco2_19.npz",
    x=params,
    y=sum_mse_xco2_19,
    nobs=len(error[0]["err19_xco2"]),
)
