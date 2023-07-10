import argparse
import sys

import numpy as np

sys.path.append("../utilities")
import os
import pickle

from folderManagement import *
from ofio import *
from plotsUtil import *

parser = argparse.ArgumentParser(description="Plot cond qoi")
parser.add_argument(
    "-sf",
    "--studyFolders",
    type=str,
    metavar="",
    required=False,
    help="studyFolder to analyze",
    nargs="+",
    default=[],
)

parser.add_argument(
    "-pv",
    "--param_value",
    type=float,
    metavar="",
    required=False,
    help="parameter value",
    nargs="+",
    default=[],
)

parser.add_argument(
    "-pn",
    "--param_name",
    type=str,
    metavar="",
    required=False,
    help="parameter name",
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

args = parser.parse_args()
figureFolder = "Figures"
figureFolder = os.path.join(figureFolder, args.figureFolder, "cond")
makeRecursiveFolder(figureFolder)
field_names = args.field_list
param_name = args.param_name
param_vals = args.param_value
study_folders = args.studyFolders
case_folders = ["circle", "side_sparger", "multiring_4"]
symbol_folders = ["o", "s", "^"]

for case_folder in case_folders:
    os.makedirs(os.path.join(figureFolder, case_folder), exist_ok=True)
for param_val in param_vals:
    os.makedirs(
        os.path.join(figureFolder, f"acrossDesign_{param_val:.2g}"),
        exist_ok=True,
    )

cond = {}

for study_folder in study_folders:
    cond[study_folder] = {}
    for case_folder in case_folders:
        print(f"Case : {study_folder}/{case_folder}")
        with open(
            os.path.join(study_folder, case_folder, "cond.pkl"), "rb"
        ) as f:
            cond[study_folder][case_folder] = pickle.load(f)


def sequencePlotShade(val_list, vert_list, listShade):
    color = "b"
    minVal = min(listShade)
    maxVal = max(listShade)
    shadeArr = (np.array(listShade) - minVal) * 0.8 / (maxVal - minVal) + 0.2
    shades = plt.cm.Blues(shadeArr)

    for ic, (val, vert) in enumerate(zip(val_list, vert_list)):
        xval = val
        yval = vert
        plt.plot(
            xval,
            yval,
            markersize=10,
            markevery=10,
            linewidth=3,
            color=shades[ic],
        )
        ax = plt.gca()
        ax.set_ylim([0, 7])


for case_folder in case_folders:
    for field_name in field_names:
        fig = plt.figure()
        val_list = [
            cond[study_folder][case_folder][field_name]["val"]
            for study_folder in study_folders
        ]
        vert_list = [
            cond[study_folder][case_folder][field_name]["vert"]
            for study_folder in study_folders
        ]
        sequencePlotShade(
            val_list=val_list, vert_list=vert_list, listShade=param_vals
        )
        prettyLabels(field_name, "z", 14)
        plt.savefig(
            os.path.join(figureFolder, case_folder, f"{field_name}.png")
        )
        plt.close()

for study_folder, param_val in zip(study_folders, param_vals):
    for field_name in field_names:
        fig = plt.figure()
        for ic, case_folder in enumerate(case_folders):
            val = cond[study_folder][case_folder][field_name]["val"]
            vert = cond[study_folder][case_folder][field_name]["vert"]
            plt.plot(
                val,
                vert,
                symbol_folders[ic],
                markersize=10,
                markevery=10,
                linewidth=3,
            )
            ax = plt.gca()
            ax.set_ylim([0, 7])
        plt.savefig(
            os.path.join(
                figureFolder,
                f"acrossDesign_{param_val:.2g}",
                f"{field_name}.png",
            )
        )
        plt.close()
