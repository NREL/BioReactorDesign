import argparse
import sys

import numpy as np

sys.path.append("../utilities")
import os
import pickle

from folderManagement import *
from ofio import *
from prettyPlot.plotting import plt, pretty_labels, pretty_legend

from bird.utilities.label_plot import label_conv

parser = argparse.ArgumentParser(description="Compare Qoi")
parser.add_argument(
    "-df",
    "--dataFiles",
    type=str,
    metavar="",
    required=False,
    help="data files to plot",
    nargs="+",
    default=[],
)

parser.add_argument(
    "-m",
    "--mode",
    type=str,
    metavar="",
    required=False,
    help="plot mode",
    default="1D",
)
parser.add_argument(
    "-xl",
    "--xlabel",
    type=str,
    metavar="",
    required=False,
    help="label name for x",
    default=None,
)
parser.add_argument(
    "-yl",
    "--ylabel",
    type=str,
    metavar="",
    required=False,
    help="label name for y",
    default=None,
)
parser.add_argument(
    "-l",
    "--labels",
    type=str,
    metavar="",
    required=False,
    help="curve labels",
    nargs="+",
    default=[],
)

parser.add_argument(
    "-ff",
    "--figureFolder",
    type=str,
    metavar="",
    required=False,
    help="figureFolder",
    default=None,
)
parser.add_argument(
    "-n",
    "--filename",
    type=str,
    metavar="",
    required=False,
    help="filename of figure",
    default=None,
)


args = parser.parse_args()
figureFolder = os.path.join("Figures", args.figureFolder)
dataFiles = args.dataFiles
mode = args.mode

makeRecursiveFolder(figureFolder)
data_structs = [np.load(dataFile) for dataFile in dataFiles]

symbol_list = ["o", "s", "^"]

if mode.lower() == "1d":
    fig = plt.figure()
    for istruct, data_struct in enumerate(data_structs):
        plt.plot(
            data_struct["x"],
            data_struct["y"],
            symbol_list[istruct],
            markersize=10,
            color="k",
            label=args.labels[istruct],
        )
    pretty_legend()
    pretty_labels(label_conv(args.xlabel), label_conv(args.ylabel), 14)

plt.savefig(os.path.join(figureFolder, f"{args.filename}.png"))
plt.savefig(os.path.join(figureFolder, f"{args.filename}.eps"))
