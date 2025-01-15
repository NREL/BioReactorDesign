import argparse

import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from prettyPlot.plotting import *

parser = argparse.ArgumentParser(description="Plot the generated data")
parser.add_argument(
    "-ter",
    "--ternary",
    action="store_true",
    help="Ternary breakup and coalescence case",
)
args, unknown = parser.parse_known_args()

ternary = args.ternary

if ternary:
    target = np.load("target_n_3_br_0.5_cr_0.5.npz")
else:
    target = np.load("target_n_2_br_1.6_cr_2.0.npz")

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
plt.plot(beff_fact, ceff, "o", color="k", label="Successful sim.")
plt.plot(beff_fact_fail, ceff_fail, "o", color="r", label="Runaway sim.")
pretty_labels("Bf [-]", "Cr [Hz]", grid=False)
pretty_legend(loc="lower left")
plt.savefig("succ_fail.png")
plt.savefig("succ_fail.eps")

# Color by beff_fact
# Normalize the colors array to map it to the colormap
beff_fact_arr = np.array(beff_fact)
# norm = Normalize(vmin=beff_fact_arr.min(), vmax=beff_fact_arr.max())
norm = Normalize(vmin=0.8, vmax=1.1)
cmap = plt.cm.viridis  # Choose a colormap
color_map = ScalarMappable(norm=norm, cmap=cmap)

fig = plt.figure()
icount = 0
plt.plot(A["x"], A["y"][0, :], color="k", label="Forward model")
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
    label="Target data",
)
ax = plt.gca()
cbarticks = [0.8, 1.1]
cbarticks_labels = ["0.8", "1.1"]
cbar = plt.colorbar(
    color_map, ax=ax, orientation="vertical", ticks=cbarticks
)  # or 'horizontal'
cbar.set_label("Bf [-]", labelpad=-10)
if cbarticks_labels is not None:
    cbar.ax.set_yticklabels(cbarticks_labels)
cbar_ax = cbar.ax
text = cbar_ax.yaxis.label
font = matplotlib.font_manager.FontProperties(
    family="serif", weight="bold", size=14
)
text.set_font_properties(font)
for l in cbar_ax.yaxis.get_ticklabels():
    l.set_weight("bold")
    l.set_family("serif")
    l.set_fontsize(12)
pretty_labels("bubble diameter [m]", "PDF", 14, grid=False)
pretty_legend()
if ternary:
    plt.savefig("cmap_bf_ternary.png")
    plt.savefig("cmap_bf_ternary.eps")
else:
    plt.savefig("cmap_bf_binary.png")
    plt.savefig("cmap_bf_binary.eps")


ceff_arr = np.array(ceff)
# norm = Normalize(vmin=ceff_arr.min(), vmax=ceff_arr.max())
norm = Normalize(vmin=0.02, vmax=2.0)
cmap = plt.cm.viridis  # Choose a colormap
color_map = ScalarMappable(norm=norm, cmap=cmap)

fig = plt.figure()
icount = 0
plt.plot(A["x"], A["y"][0, :], color="k", label="Forward model")
for i in range(nsim):
    if abs(np.mean(A["y"][i, :]) - 1) > 1e-6:
        plt.plot(
            A["x"], A["y"][i, :], color=color_map.to_rgba(ceff_arr[icount])
        )
        icount += 1
im = plt.plot(
    target["x"],
    target["y"],
    "^",
    markersize=10,
    markeredgecolor="k",
    markerfacecolor="r",
    label="Target data",
)
ax = plt.gca()

cbarticks = [0.02, 2.0]
cbarticks_labels = ["0.02", "2.0"]
cbar = plt.colorbar(
    color_map, ax=ax, orientation="vertical", ticks=cbarticks
)  # or 'horizontal'
cbar.set_label("Cr [Hz]", labelpad=-10)
if cbarticks_labels is not None:
    cbar.ax.set_yticklabels(cbarticks_labels)
cbar_ax = cbar.ax
text = cbar_ax.yaxis.label
font = matplotlib.font_manager.FontProperties(
    family="serif", weight="bold", size=14
)
text.set_font_properties(font)
for l in cbar_ax.yaxis.get_ticklabels():
    l.set_weight("bold")
    l.set_family("serif")
    l.set_fontsize(12)
pretty_labels("bubble diameter [m]", "PDF", 14, grid=False)
pretty_legend()
if ternary:
    plt.savefig("cmap_cr_ternary.png")
    plt.savefig("cmap_cr_ternary.eps")
else:
    plt.savefig("cmap_cr_binary.png")
    plt.savefig("cmap_cr_binary.eps")


plt.show()
