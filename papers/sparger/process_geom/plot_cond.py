import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from prettyPlot.plotting import plt, pretty_labels

from bird.utilities.ofio import *

from .label_plot import label_conv


def getManyFolders(rootFolder, prefix="flat_donut"):
    # Read Time
    fold_tmp = os.listdir(rootFolder)
    fold_num = []
    # remove non floats
    for i, entry in reversed(list(enumerate(fold_tmp))):
        if not entry.startswith(prefix) or entry.endswith("N2"):
            a = fold_tmp.pop(i)
            # print('removed ', a)
    for entry in fold_tmp:
        num = re.findall(r"\d+", entry)
        if len(num) > 1:
            msg = f"Cannot find num of folder {entry}."
            msg += "\nDo not trust the spearman stat"
            logger.warning(msg)
        else:
            fold_num.append(int(num[0]))

    sortedFold = [
        x for _, x in sorted(zip(fold_num, fold_tmp), key=lambda pair: pair[0])
    ]
    return sortedFold


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
Path(figureFolder).mkdir(parents=True, exist_ok=True)
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
    # shades = plt.cm.Blues(shadeArr)
    shades = plt.cm.Greys(shadeArr)

    for ic, c in enumerate(seq):
        xval = seq[c][fieldName]["val"]
        yval = seq[c][fieldName]["vert"]
        ind = np.argwhere((yval >= 0.5) & (yval < 7))
        plt.plot(
            xval[ind],
            yval[ind],
            markersize=10,
            markevery=10,
            linewidth=3,
            color=shades[ic],
        )
        ax = plt.gca()
        # ax.set_ylim([0, 7])


for field_name in field_names:
    fig = plt.figure()
    sequencePlotShade(cond, params[param_name][ind_keep], field_name)
    pretty_labels(label_conv(field_name), "y [m]", 14)
    plt.savefig(os.path.join(figureFolder, param_name, f"{field_name}.png"))
    plt.savefig(os.path.join(figureFolder, param_name, f"{field_name}.eps"))
    plt.close()
