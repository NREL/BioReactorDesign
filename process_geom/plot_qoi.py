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
    default=None,
)
parser.add_argument(
    "-cp",
    "--casePrefix",
    type=str,
    metavar="",
    required=False,
    help="case folder prefix to analyze",
    default=None,
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
    "-pl",
    "--param_list",
    type=str,
    metavar="",
    required=False,
    help="parameters to analyze",
    nargs="+",
    default=["width"],
)
parser.add_argument(
    "-cfe",
    "--case_folders_exclude",
    type=str,
    metavar="",
    required=False,
    help="case folders to exclude",
    nargs="+",
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
    "-p",
    "--paramFile",
    type=str,
    metavar="",
    required=True,
    help="parameter being varied",
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
param_file = args.paramFile
study_folder = args.studyFolder
param_names = args.param_list
case_folder_exclude = args.case_folders_exclude
for param_name in param_names:
    os.makedirs(os.path.join(figure_qoi_Folder, param_name), exist_ok=True)

params = np.load(os.path.join(study_folder, param_file))
case_folders = getManyFolders(study_folder, prefix=args.casePrefix)
for folder in case_folders:
    if not os.path.isfile(os.path.join(study_folder, folder, "qoi.pkl")):
        if folder not in case_folder_exclude:
            case_folder_exclude += [folder]
ind_exclude = []
for folder in case_folder_exclude:
    if folder in case_folders:
        ind_exclude.append(case_folders.index(folder))
ind_keep = list(set(list(range(len(case_folders)))).difference(ind_exclude))
case_folders_final = []
for folder in case_folders:
    if folder not in case_folder_exclude:
        case_folders_final.append(folder)


qoi = {}

for case_folder in case_folders_final:
    print(f"Case : {case_folder}")
    with open(os.path.join(study_folder, case_folder, "qoi.pkl"), "rb") as f:
        qoi[case_folder] = pickle.load(f)

for var_name in var_names:
    for param_name in param_names:
        fig = plt.figure()
        var_val = [
            qoi[case_folder][var_name] for case_folder in case_folders_final
        ]
        plt.plot(params[param_name][ind_keep], var_val, "o", color="k")
        prettyLabels(param_name, var_name, 14)
        plt.savefig(
            os.path.join(figure_qoi_Folder, param_name, f"{var_name}.png")
        )
        plt.close()

qoi_conv = {}

for case_folder in case_folders_final:
    print(f"Conv Case : {case_folder}")
    with open(
        os.path.join(study_folder, case_folder, "qoi_conv.pkl"), "rb"
    ) as f:
        qoi_conv[case_folder] = pickle.load(f)

for var_name in var_names:
    for param_name in param_names:
        fig = plt.figure()
        for case_folder in case_folders_final:
            plt.plot(
                qoi_conv[case_folder][var_name]["x"],
                qoi_conv[case_folder][var_name]["y"],
                linewidth=2,
                color="k",
            )
        for case_folder in case_folders_final:
            plt.plot(
                qoi_conv[case_folder][var_name]["x"][-1],
                qoi_conv[case_folder][var_name]["y"][-1],
                "*",
                markersize=15,
                color="k",
            )
        prettyLabels("t [s]", var_name, 14)
        plt.savefig(os.path.join(figure_qoiConv_Folder, f"{var_name}.png"))
        plt.close()
