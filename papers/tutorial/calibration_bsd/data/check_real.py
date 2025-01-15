import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from prettyPlot.plotting import *

target = np.load("target.npz")
A = np.load("dataset.npz")
nsim = A["y"].shape[0]
print(f"nsim = {nsim}")
ceff = []
beff_fact = []
ceff_fail = []
beff_fact_fail = []
for i in range(nsim):
    if abs(np.mean(A["y"][i, :]) - 1) > 1e-6:
        print(
            f"beff_fact : {A['beff_fact'][i]:.2f}, ceff : {A['ceff'][i]:.2f}"
        )
        ceff.append(A["ceff"][i])
        beff_fact.append(A["beff_fact"][i])
    else:
        ceff_fail.append(A["ceff"][i])
        beff_fact_fail.append(A["beff_fact"][i])
fig = plt.figure()
plt.plot(beff_fact, ceff, "o", color="k")
plt.plot(beff_fact_fail, ceff_fail, "o", color="r")
pretty_labels("beff_fact", "ceff")


# Color by beff_fact
# Normalize the colors array to map it to the colormap
beff_fact_arr = np.array(beff_fact)
norm = Normalize(vmin=beff_fact_arr.min(), vmax=beff_fact_arr.max())
cmap = plt.cm.viridis  # Choose a colormap
color_map = ScalarMappable(norm=norm, cmap=cmap)

fig = plt.figure()
icount = 0
for i in range(nsim):
    if abs(np.mean(A["y"][i, :]) - 1) > 1e-6:
        plt.plot(
            A["x"],
            A["y"][i, :],
            color=color_map.to_rgba(beff_fact_arr[icount]),
        )
        icount += 1
plt.plot(
    target["x"],
    target["y"],
    "^",
    markersize=10,
    markeredgecolor="k",
    markerfacecolor="r",
)
pretty_labels("x", "y", 14, title="beff_fact")


ceff_arr = np.array(ceff)
norm = Normalize(vmin=ceff_arr.min(), vmax=ceff_arr.max())
cmap = plt.cm.viridis  # Choose a colormap
color_map = ScalarMappable(norm=norm, cmap=cmap)

fig = plt.figure()
icount = 0
for i in range(nsim):
    if abs(np.mean(A["y"][i, :]) - 1) > 1e-6:
        plt.plot(
            A["x"], A["y"][i, :], color=color_map.to_rgba(ceff_arr[icount])
        )
        icount += 1
plt.plot(
    target["x"],
    target["y"],
    "^",
    markersize=10,
    markeredgecolor="k",
    markerfacecolor="r",
)
pretty_labels("x", "y", 14, title="ceff")


plt.show()
