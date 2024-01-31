import argparse
import sys

import numpy as np

sys.path.append("../utilities")
import os
import pickle

from folderManagement import *
from ofio import *
from plotsUtil import *

parser = argparse.ArgumentParser(description="Plot conditional means")
parser.add_argument(
    "-cf",
    "--caseFolders",
    type=str,
    metavar="",
    required=False,
    help="caseFolder to analyze",
    nargs="+",
    default=None,
)
parser.add_argument(
    "-fl",
    "--field_list",
    type=str,
    metavar="",
    required=False,
    help="fields to analyze",
    nargs="+",
    default=[
        "CO.gas",
        "CO.liquid",
        "CO2.gas",
        "CO2.liquid",
        "H2.gas",
        "H2.liquid",
        "alpha.gas",
        "d.gas",
    ],
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
    "--names",
    type=str,
    metavar="",
    required=False,
    help="names of cases",
    nargs="+",
    default=None,
)


def sequencePlot(cond, case_names, folder_names, field_name, symbList):
    for ic, (case_name, folder_name) in enumerate(
        zip(case_names, folder_names)
    ):
        label = ""
        if ic == 0:
            label = case_names[ic]
        plt.plot(
            cond[folder_name][field_name]["val"],
            cond[folder_name][field_name]["vert"],
            symbList[ic],
            markersize=10,
            markevery=10,
            linewidth=3,
            color="k",
            label=label,
        )


args = parser.parse_args()
figureFolder = "Figures"
figureFolder = os.path.join(figureFolder, args.figureFolder)
makeRecursiveFolder(figureFolder)
field_names = args.field_list
case_names = args.names
case_folders = args.caseFolders

assert len(case_names) == len(case_folders)

symbList = ["-", "-d", "-^", "-.", "-s", "-o", "-+"]
if len(case_names) > len(symbList):
    print(
        f"ERROR: too many cases ({len(case_names)}), reduce number of case to {len(symbList)} or add symbols"
    )
    sys.exit()


cond = {}

for case_folder in case_folders:
    print(f"Case : {case_folder}")
    with open(os.path.join(case_folder, "cond.pkl"), "rb") as f:
        cond[case_folder] = pickle.load(f)


for field_name in field_names:
    fig = plt.figure()
    sequencePlot(cond, case_names, case_folders, field_name, symbList)
    if field_name == "alpha.gas":
        plot_name = "gasHoldup"
    else:
        plot_name = field_name
    prettyLabels(plot_name, "y [m]", 14)
    plt.savefig(os.path.join(figureFolder, f"{plot_name}.png"))
    plt.close()
