import argparse
import sys

import numpy as np

sys.path.append("../utilities")
import os

from folderManagement import *
from ofio import *
from plotsUtil import *

parser = argparse.ArgumentParser(description="Case folder")
parser.add_argument(
    "-rd",
    "--refData",
    type=str,
    metavar="",
    required=True,
    help="reference data",
    default=None,
)
parser.add_argument(
    "-f17",
    "--caseFolder17",
    type=str,
    metavar="",
    required=True,
    help="caseFolder17 to analyze",
    default=None,
)
parser.add_argument(
    "-f19",
    "--caseFolder19",
    type=str,
    metavar="",
    required=True,
    help="caseFolder19 to analyze",
    default=None,
)
parser.add_argument(
    "-ff",
    "--figureFolder",
    type=str,
    metavar="",
    required=True,
    help="figureFolder",
    default=None,
)

args = parser.parse_args()

figureFolder = "Figures"
figureFolder = os.path.join(figureFolder, args.figureFolder)
makeRecursiveFolder(figureFolder)

conv17 = np.load(os.path.join(args.caseFolder17, "convergence_gh.npz"))
conv19 = np.load(os.path.join(args.caseFolder19, "convergence_gh.npz"))
sim17 = np.load(os.path.join(args.caseFolder17, "observations.npz"))
sim19 = np.load(os.path.join(args.caseFolder19, "observations.npz"))
refData = np.load(args.refData)


def convertToMol(yCO2):
    MN2 = 0.028
    MCO2 = 0.044
    return MN2 * yCO2 / (MCO2 + MN2 * yCO2 - MCO2 * yCO2)


fig = plt.figure()
plt.plot(conv17["time"], conv17["gh"], linewidth=3, color="k", label="Sim17")
plt.plot(conv19["time"], conv19["gh"], linewidth=3, color="b", label="Sim19")
plotLegend()
prettyLabels("t [s]", "Gas Holdup", 14)
plt.savefig(os.path.join(figureFolder, "conv.png"))
plt.close()

fig = plt.figure()
plt.plot(
    refData["gh_exp17"][:, 0],
    refData["gh_exp17"][:, 1],
    "o",
    color="b",
    label="exp. case 17",
)
plt.plot(
    refData["gh_exp19"][:, 0],
    refData["gh_exp19"][:, 1],
    "o",
    color="r",
    label="exp. case 19",
)
plt.plot(
    refData["gh_exp17_ngu"][:, 0],
    refData["gh_exp17_ngu"][:, 1],
    color="b",
    label="Ngu et al.",
)
plt.plot(
    refData["gh_exp19_ngu"][:, 0],
    refData["gh_exp19_ngu"][:, 1],
    color="r",
)
plt.plot(
    refData["gh_exp17_hassaniga"][:, 0],
    refData["gh_exp17_hassaniga"][:, 1],
    "--",
    color="b",
    label="Hissanaga et al.",
)
plt.plot(
    refData["gh_exp19_hassaniga"][:, 0],
    refData["gh_exp19_hassaniga"][:, 1],
    "--",
    color="r",
)
plt.plot(sim17["gh"], sim17["z"], linewidth=3, color="b", label="This work")
plt.plot(sim19["gh"], sim19["z"], linewidth=3, color="r")
# plotLegend()
prettyLabels("Gas Holdup", "z [m]", 14)
plt.savefig(os.path.join(figureFolder, "gh.png"))
plt.savefig(os.path.join(figureFolder, "gh.eps"))
plt.close()

fig = plt.figure()
plt.plot(
    refData["xco2_exp17"][:, 0],
    refData["xco2_exp17"][:, 1],
    "o",
    color="b",
    label="exp. case 17",
)
plt.plot(
    refData["xco2_exp19"][:, 0],
    refData["xco2_exp19"][:, 1],
    "o",
    color="r",
    label="exp. case 19",
)
plt.plot(
    refData["xco2_exp17_ngu"][:, 0],
    refData["xco2_exp17_ngu"][:, 1],
    color="b",
    label="Ngu et al.",
)
plt.plot(
    refData["xco2_exp19_ngu"][:, 0], refData["xco2_exp19_ngu"][:, 1], color="r"
)
plt.plot(
    refData["xco2_exp17_hassaniga"][:, 0],
    refData["xco2_exp17_hassaniga"][:, 1],
    "--",
    color="b",
    label="Hissanaga et al.",
)
plt.plot(
    refData["xco2_exp19_hassaniga"][:, 0],
    refData["xco2_exp19_hassaniga"][:, 1],
    "--",
    color="r",
)
plt.plot(
    convertToMol(sim17["co2"]),
    sim17["z"],
    linewidth=3,
    color="b",
    label="This work",
)
plt.plot(convertToMol(sim19["co2"]), sim19["z"], linewidth=3, color="r")
# plotLegend()
prettyLabels(r"$X_{CO2}$", "z [m]", 14)
plt.savefig(os.path.join(figureFolder, "co2.png"))
plt.savefig(os.path.join(figureFolder, "co2.eps"))
plt.close()
