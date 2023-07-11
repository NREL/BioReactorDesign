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
figureFolder = os.path.join(figureFolder, args.figureFolder, "cond")
makeRecursiveFolder(figureFolder)
field_names = args.field_list
param_file = args.paramFile
study_folder = args.studyFolder
param_names = args.param_list
case_folder_exclude = args.case_folders_exclude
for param_name in param_names:
    os.makedirs(os.path.join(figureFolder, param_name), exist_ok=True)

params = np.load(os.path.join(study_folder, param_file))
case_folders = getManyFolders(study_folder, prefix=args.casePrefix)
for folder in case_folders:
    if not os.path.isfile(os.path.join(study_folder, folder, "cond.pkl")):
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


cond = {}

for case_folder in case_folders_final:
    print(f"Case : {case_folder}")
    with open(os.path.join(study_folder, case_folder, "cond.pkl"), "rb") as f:
        cond[case_folder] = pickle.load(f)


def sequencePlotShade(seq, listShade, fieldName):
    color = "b"
    minVal = min(listShade)
    maxVal = max(listShade)
    shadeArr = (np.array(listShade) - minVal) * 0.8 / (maxVal - minVal) + 0.2
    shades = plt.cm.Blues(shadeArr)

    for ic, c in enumerate(seq):
        xval = seq[c][fieldName]["val"]
        yval = seq[c][fieldName]["vert"]
        ind = np.argwhere((yval>=0.5) & (yval<7))
        plt.plot(
            xval[ind],
            yval[ind],
            markersize=10,
            markevery=10,
            linewidth=3,
            color=shades[ic],
        )
        ax = plt.gca()
        #ax.set_ylim([0, 7])


for field_name in field_names:
    fig = plt.figure()
    sequencePlotShade(cond, params[param_name][ind_keep], field_name)
    prettyLabels(field_name, "z", 14)
    plt.savefig(os.path.join(figureFolder, param_name, f"{field_name}.png"))
    plt.close()
