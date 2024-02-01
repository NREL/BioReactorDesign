import argparse
import sys

import numpy as np

sys.path.append("../utilities")
import os
import pickle

from folderManagement import *
from ofio import *
from prettyPlot.plotting import plt, pretty_labels

from brd.utilities.label_plot import label_conv

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
    "-vmin",
    "--vmin",
    type=float,
    metavar="",
    required=False,
    help="min to plot",
    nargs="+",
    default=[],
)
parser.add_argument(
    "-vmax",
    "--vmax",
    type=float,
    metavar="",
    required=False,
    help="max to plot",
    nargs="+",
    default=[],
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
npFolder = "Data"
figure_qoi_folder = os.path.join(figureFolder, args.figureFolder, "qoi")
figure_qoiNP_folder = os.path.join(npFolder, args.figureFolder, "qoi")
figure_qoiConv_folder = os.path.join(
    figureFolder, args.figureFolder, "qoi_conv"
)


makeRecursiveFolder(figure_qoi_folder)
makeRecursiveFolder(figure_qoiNP_folder)
makeRecursiveFolder(figure_qoiConv_folder)
var_names = args.var_list
vmin = args.vmin
vmax = args.vmax

use_auto_scale = True
if len(vmin) == len(var_names) and len(vmax) == len(var_names):
    use_auto_scale = False

param_file = args.paramFile
study_folder = args.studyFolder
param_names = args.param_list
plot_2d_param_space = False
if len(param_names) == 2:
    plot_2d_param_space = True
case_folder_exclude = args.case_folders_exclude
for param_name in param_names:
    os.makedirs(os.path.join(figure_qoi_folder, param_name), exist_ok=True)
    os.makedirs(os.path.join(figure_qoiNP_folder, param_name), exist_ok=True)

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

for ivar, var_name in enumerate(var_names):
    for param_name in param_names:
        fig = plt.figure()
        var_val = [
            qoi[case_folder][var_name] for case_folder in case_folders_final
        ]
        plt.plot(params[param_name][ind_keep], var_val, "o", color="k")
        pretty_labels(label_conv(param_name), label_conv(var_name), 14)
        if not use_auto_scale:
            ax = plt.gca()
            ax.set_ylim([vmin[ivar], vmax[ivar]])
        plt.savefig(
            os.path.join(figure_qoi_folder, param_name, f"{var_name}.png")
        )
        plt.savefig(
            os.path.join(figure_qoi_folder, param_name, f"{var_name}.eps")
        )
        np.savez(
            os.path.join(figure_qoiNP_folder, param_name, f"{var_name}"),
            x=params[param_name][ind_keep],
            y=var_val,
        )
        plt.close()

if plot_2d_param_space:
    for var_name in var_names:
        fig = plt.figure()
        var_val = [
            qoi[case_folder][var_name] for case_folder in case_folders_final
        ]
        # avoid white dots
        var_val_scaled = [val * 0.995 for val in var_val]
        plt.scatter(
            params[param_names[0]][ind_keep],
            params[param_names[1]][ind_keep],
            s=100,
            c=var_val_scaled,
            cmap="gray",
            vmin=np.amin(var_val),
            vmax=np.amax(var_val),
        )
        cbar = plt.colorbar()
        pretty_labels(
            label_conv(param_names[0]),
            label_conv(param_names[1]),
            title=label_conv(var_name),
            fontsize=14,
            grid=False,
        )
        for l in cbar.ax.yaxis.get_ticklabels():
            l.set_weight("bold")
            l.set_family("serif")
            l.set_fontsize(12)
        # cax = cbar.ax
        # text = cax.yaxis.label
        # font = matplotlib.font_manager.FontProperties(
        #    family="times new roman", weight="bold", size=14
        # )
        # text.set_font_properties(font)
        plt.savefig(os.path.join(figure_qoi_folder, f"{var_name}_2d.png"))
        plt.savefig(os.path.join(figure_qoi_folder, f"{var_name}_2d.eps"))
        np.savez(
            os.path.join(figure_qoiNP_folder, f"{var_name}_2d"),
            x=params[param_names[0]][ind_keep],
            y=params[param_names[1]][ind_keep],
            z=var_val,
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
        pretty_labels("t [s]", label_conv(var_name), 14)
        plt.savefig(os.path.join(figure_qoiConv_folder, f"{var_name}.png"))
        plt.savefig(os.path.join(figure_qoiConv_folder, f"{var_name}.eps"))
        plt.close()
