import os
import pickle
import sys

import numpy as np

sys.path.append("util")
import matplotlib.pyplot as plt
from prettyPlot.plotting import plt, pretty_labels
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
mse_gh = np.zeros(nSim)
mse_xco2 = np.zeros(nSim)
accuracy_gh_17 = np.zeros(nSim)
accuracy_xco2_17 = np.zeros(nSim)
accuracy_gh_19 = np.zeros(nSim)
accuracy_xco2_19 = np.zeros(nSim)
params = np.zeros((nSim, nParams))


iSim = 0
for indSim in error:
    err = error[indSim]
    if limiter(err):
        continue
    accuracy_gh_17[iSim] = np.mean(abs(err["err17_gh"]))
    accuracy_xco2_17[iSim] = np.mean(abs(err["err17_xco2"]))
    accuracy_gh_19[iSim] = np.mean(abs(err["err19_gh"]))
    accuracy_xco2_19[iSim] = np.mean(abs(err["err19_xco2"]))
    mse_gh[iSim] = np.sum(err["err17_gh"] ** 2) + np.sum(err["err19_gh"] ** 2)
    mse_xco2[iSim] = np.sum(err["err17_xco2"] ** 2) + np.sum(
        err["err19_xco2"] ** 2
    )
    for iname, name in enumerate(var_names):
        params[iSim, iname] = err[name]
    iSim += 1

accuracy_gh = accuracy_gh_17 + accuracy_gh_19
accuracy_xco2 = accuracy_xco2_17 + accuracy_xco2_19
accuracy_17 = accuracy_gh_17 + accuracy_xco2_17
accuracy_19 = accuracy_gh_19 + accuracy_xco2_19
tot_accuracy = accuracy_gh + accuracy_xco2

minInd = np.argmin(tot_accuracy)
print("BEST")
for iname, name in enumerate(var_names):
    print(f"\t{name} : {params[minInd, iname]}")
minInd = np.argmin(accuracy_gh)
print("BEST GH")
for iname, name in enumerate(var_names):
    print(f"\t{name} : {params[minInd, iname]}")
minInd = np.argmin(accuracy_xco2)
print("BEST XCO2")
for iname, name in enumerate(var_names):
    print(f"\t{name} : {params[minInd, iname]}")
minInd = np.argmin(accuracy_17)
print("BEST 17")
for iname, name in enumerate(var_names):
    print(f"\t{name} : {params[minInd, iname]}")
minInd = np.argmin(accuracy_19)
print("BEST 19")
for iname, name in enumerate(var_names):
    print(f"\t{name} : {params[minInd, iname]}")


figureFolder = "Figures"
os.makedirs(figureFolder, exist_ok=True)


for iname, name in enumerate(var_names):
    fig = plt.figure()
    plt.plot(params[:, iname], tot_accuracy, "o", color="k")
    pretty_labels(name, "accuracy", 14)
    plt.savefig(os.path.join(figureFolder, "acc_" + name + ".png"))
    plt.close()
for iname, name in enumerate(var_names):
    fig = plt.figure()
    plt.plot(params[:, iname], accuracy_gh, "o", color="k")
    pretty_labels(name, "accuracy_gh", 14)
    plt.savefig(os.path.join(figureFolder, "acc_gh_" + name + ".png"))
    plt.close()
for iname, name in enumerate(var_names):
    fig = plt.figure()
    plt.plot(params[:, iname], accuracy_xco2, "o", color="k")
    pretty_labels(name, "accuracy_xco2", 14)
    plt.savefig(os.path.join(figureFolder, "acc_xco2_" + name + ".png"))
    plt.close()
for iname, name in enumerate(var_names):
    fig = plt.figure()
    plt.plot(params[:, iname], accuracy_17, "o", color="k")
    pretty_labels(name, "accuracy_17", 14)
    plt.savefig(os.path.join(figureFolder, "acc_17_" + name + ".png"))
    plt.close()
for iname, name in enumerate(var_names):
    fig = plt.figure()
    plt.plot(params[:, iname], accuracy_19, "o", color="k")
    pretty_labels(name, "accuracy_19", 14)
    plt.savefig(os.path.join(figureFolder, "acc_19_" + name + ".png"))
    plt.close()


# plt.show()
