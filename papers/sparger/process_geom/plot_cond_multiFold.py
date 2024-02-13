import argparse
import sys

import numpy as np

sys.path.append("../utilities")
import os
import pickle

from folderManagement import *
from ofio import *
from prettyPlot.plotting import plt, pretty_labels

from bird.utilities.label_plot import label_conv

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
case_names = ["Circle", "Side sparger", "Multiring"]
symbol_folders = ["-o", "-s", "-^"]

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
    # shades = plt.cm.Blues(shadeArr)
    shades = plt.cm.Greys(shadeArr)

    for ic, (val, vert) in enumerate(zip(val_list, vert_list)):
        xval = val
        yval = vert
        ind = np.argwhere((yval >= 0.5) & (yval < 7))
        plt.plot(
            xval[ind],
            yval[ind],
            # markersize=10,
            # markevery=10,
            linewidth=3,
            color=shades[ic],
        )
        ax = plt.gca()
        # ax.set_ylim([0, 7])


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
        pretty_labels(label_conv(field_name), "y [m]", 14, title=" ")
        plt.savefig(
            os.path.join(figureFolder, case_folder, f"{field_name}.png")
        )
        plt.savefig(
            os.path.join(figureFolder, case_folder, f"{field_name}.eps")
        )
        plt.close()

for study_folder, param_val in zip(study_folders, param_vals):
    for field_name in field_names:
        fig = plt.figure()
        for ic, case_folder in enumerate(case_folders):
            val = cond[study_folder][case_folder][field_name]["val"]
            vert = cond[study_folder][case_folder][field_name]["vert"]
            ind = np.argwhere((vert >= 0.5) & (vert < 7))
            plt.plot(
                val[ind],
                vert[ind],
                symbol_folders[ic],
                markersize=10,
                color="k",
                linewidth=3,
                label=case_names[ic],
            )
            ax = plt.gca()
            # ax.set_ylim([0, 7])
            pretty_labels(label_conv(field_name), "y [m]", 14, title=" ")
            pretty_legend()
        plt.savefig(
            os.path.join(
                figureFolder,
                f"acrossDesign_{param_val:.2g}",
                f"{field_name}.png",
            )
        )
        plt.savefig(
            os.path.join(
                figureFolder,
                f"acrossDesign_{param_val:.2g}",
                f"{field_name}.eps",
            )
        )
        plt.close()
