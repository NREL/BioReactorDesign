import argparse
import sys

import numpy as np

sys.path.append("util")
import os
import pickle

from folderManagement import *
from ofio import *
from plotsUtil import *

parser = argparse.ArgumentParser(description="Plot Qoi")
parser.add_argument(
    "-sf",
    "--studyFolder",
    type=str,
    metavar="",
    required=False,
    help="studyFolder to analyze",
    nargs="+",
    default=[],
)

parser.add_argument(
    "-vl",
    "--var_list",
    type=str,
    metavar="",
    required=False,
    help="variables to analyze",
    nargs="+",
    default=[
        "GH",
        "GH_height",
        "d",
        "CO2_liq",
        "CO_liq",
        "H2_liq",
        "kla_CO2",
        "kla_CO",
        "kla_H2",
    ],
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
figure_qoi_Folder = os.path.join(figureFolder, args.figureFolder, "qoi")
figure_qoiConv_Folder = os.path.join(
    figureFolder, args.figureFolder, "qoi_conv"
)
makeRecursiveFolder(figure_qoi_Folder)
makeRecursiveFolder(figure_qoiConv_Folder)
var_names = args.var_list
param_name = args.param_name
param_vals = args.param_value
study_folders = args.studyFolder
case_folders = ["circle", "side_sparger", "multiring_4"]

for case_folder in case_folders:
    os.makedirs(os.path.join(figure_qoi_Folder, case_folder), exist_ok=True)
    os.makedirs(
        os.path.join(figure_qoiConv_Folder, case_folder), exist_ok=True
    )
for param_val in param_vals:
    os.makedirs(
        os.path.join(figure_qoi_Folder, f"acrossDesign_{param_val:.2g}"),
        exist_ok=True,
    )


qoi = {}

for study_folder in study_folders:
    qoi[study_folder] = {}
    for case_folder in case_folders:
        print(f"Case : {study_folder}/{case_folder}")
        with open(
            os.path.join(study_folder, case_folder, "qoi.pkl"), "rb"
        ) as f:
            qoi[study_folder][case_folder] = pickle.load(f)

for case_folder in case_folders:
    for var_name in var_names:
        fig = plt.figure()
        var_val = [
            qoi[study_folder][case_folder][var_name]
            for study_folder in study_folders
        ]
        plt.plot(param_vals, var_val, "o", color="k")
        prettyLabels(param_name, var_name, 14)
        plt.savefig(
            os.path.join(figure_qoi_Folder, case_folder, f"{var_name}.png")
        )
        plt.close()

for study_folder, param_val in zip(study_folders, param_vals):
    for var_name in var_names:
        fig = plt.figure()
        var_val = [
            qoi[study_folder][case_folder][var_name]
            for case_folder in case_folders
        ]
        plot_bar_names(case_folders, var_val)
        plt.savefig(
            os.path.join(
                figure_qoi_Folder,
                f"acrossDesign_{param_val:.2g}",
                f"{var_name}.png",
            )
        )
        plt.close()

qoi_conv = {}

for study_folder in study_folders:
    qoi_conv[study_folder] = {}
    for case_folder in case_folders:
        print(f"Case : {study_folder}/{case_folder}")
        with open(
            os.path.join(study_folder, case_folder, "qoi_conv.pkl"), "rb"
        ) as f:
            qoi_conv[study_folder][case_folder] = pickle.load(f)

for case_folder in case_folders:
    for var_name in var_names:
        fig = plt.figure()
        for study_folder in study_folders:
            plt.plot(
                qoi_conv[study_folder][case_folder][var_name]["x"],
                qoi_conv[study_folder][case_folder][var_name]["y"],
                linewidth=2,
                color="k",
            )
        for study_folder in study_folders:
            plt.plot(
                qoi_conv[study_folder][case_folder][var_name]["x"][-1],
                qoi_conv[study_folder][case_folder][var_name]["y"][-1],
                "*",
                markersize=15,
                color="k",
            )
        prettyLabels("t [s]", var_name, 14)
        plt.savefig(
            os.path.join(figure_qoiConv_Folder, case_folder, f"{var_name}.png")
        )
        plt.close()
