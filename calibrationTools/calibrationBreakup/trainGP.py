import os
import sys

import numpy as np

sys.path.append("util")
import gpfit
from plotsUtil import *

models = [
    {"name": "gh_17", "data": "train_data_gh_17.npz"},
    {"name": "gh_19", "data": "train_data_gh_19.npz"},
    {"name": "xco2_17", "data": "train_data_xco2_17.npz"},
    {"name": "xco2_19", "data": "train_data_xco2_19.npz"},
]
var_names = ["surfaceTension", "henry", "coal_eff", "breakup_eff"]

figureFolder = "Figures"
os.makedirs(figureFolder, exist_ok=True)

for imodel, model in enumerate(models):
    data = np.load(model["data"])
    x = data["x"]
    y = data["y"]
    y_mean, y_std, gpr, like = gpfit.gpfit_simple(x, y, len(var_names))
    x_interp_1d = []
    for iname, name in enumerate(var_names):
        if name == "surfaceTension":
            x_interp_1d.append(np.linspace(0.035, 0.14, 10))
        elif name == "henry":
            x_interp_1d.append(np.linspace(0.52, 2.08, 10))
        elif name == "coal_eff":
            x_interp_1d.append(np.linspace(0.1, 10, 10))
        elif name == "breakup_eff":
            x_interp_1d.append(np.linspace(0.1, 10, 10))
    x_interp = np.meshgrid(*x_interp_1d)
    x_call_interp = np.hstack(
        (np.reshape(xx_interp, (-1, 1)) for xx_interp in x_interp)
    )
    y_interp, y_std = gpr.predict(x_call_interp, return_std=True)

    models[imodel]["amplitude"] = gpr.kernel_.get_params()["k1__constant_value"]
    models[imodel]["length_scale"] = gpr.kernel_.get_params()["k2__length_scale"]
    print(f"{model['name']} : {gpr.kernel_}")
    os.makedirs(os.path.join(figureFolder, model["name"]), exist_ok=True)

    for iname, name in enumerate(var_names):
        fig = plt.figure()
        plt.plot(x_call_interp[:, iname], y_std, "o", color="k")
        prettyLabels(name, "std GP", 14)
        plt.savefig(os.path.join(figureFolder, model["name"], "stdGP_" + name + ".png"))
        plt.close()


import pickle

# create a binary pickle file
f = open("gps.pkl", "wb")
# write the python object (dict) to pickle file
pickle.dump(models, f)
# close file
f.close()
