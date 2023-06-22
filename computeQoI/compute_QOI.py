import argparse
import sys

import numpy as np

sys.path.append("util")
import os
import pickle

from ofio import *
from bcr_util import *

parser = argparse.ArgumentParser(
    description="Compute means QoI of OpenFOAM fields"
)
parser.add_argument(
    "-f",
    "--caseFolder",
    type=str,
    metavar="",
    required=True,
    help="caseFolder to analyze",
    default=None,
)
parser.add_argument(
    "-vl",
    "--var_list",
    nargs="+",
    help="List of variables to compute",
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
    required=False,
)
parser.add_argument(
    "-avg",
    "--windowAve",
    type=int,
    metavar="",
    required=False,
    help="Window Average",
    default=1,
)
args = parser.parse_args()


case_path = args.caseFolder
var_name_list = args.var_list
time_float_sorted, time_str_sorted = getCaseTimes(case_path)
mesh_time_str = getMeshTime(case_path)
cellCentres = readMesh(
    os.path.join(case_path, f"meshCellCentres_{mesh_time_str}.obj")
)
nCells = len(cellCentres)

window_ave = min(args.windowAve, len(time_str_sorted))

variables = {}
for name in var_name_list:
    variables[name] = {}

print(f"Case : {case_path}")

for i_ave in range(window_ave):
    time_folder = time_str_sorted[-i_ave - 1]
    print(f"\tTime : {time_folder}")
    case_variables = []
    for name in var_name_list:
        localFolder = os.path.join(case_path, time_folder)
        localFolder_vol = os.path.join(case_path, mesh_time_str)
        val_dict = {}
        if name == "GH":
            var, val_dict = computeGH(localFolder, localFolder_vol, nCells, val_dict)
        elif name == "GH_height":
            var, val_dict = computeGH_height(
                localFolder, nCells, cellCentres, height_liq_base=7.0, val_dict=val_dict
            )
        elif name == "d":
            var, val_dict = computeDiam(localFolder, nCells, val_dict)
        elif name == "CO2_liq":
            var, val_dict = computeSpec_liq(
                localFolder,
                nCells,
                field_name="CO2.liquid",
                key="co2_liq",
                val_dict=val_dict,
            )
        elif name == "CO_liq":
            var, val_dict = computeSpec_liq(
                localFolder,
                nCells,
                field_name="CO.liquid",
                key="co_liq",
                val_dict=val_dict,
            )
        elif name == "H2_liq":
            var, val_dict = computeSpec_liq(
                localFolder,
                nCells,
                field_name="H2.liquid",
                key="h2_liq",
                val_dict=val_dict,
            )
        elif name == "kla_CO":
            var, val_dict = computeSpec_kla(
                localFolder, nCells, key_suffix="co", val_dict=val_dict
            )
        elif name == "kla_CO2":
            var, val_dict = computeSpec_kla(
                localFolder, nCells, key_suffix="co2", val_dict=val_dict
            )
        elif name == "kla_H2":
            var, val_dict = computeSpec_kla(
                localFolder, nCells, key_suffix="h2", val_dict=val_dict
            )
        else:
            sys.exit(f"ERROR: unknown variable {name}")

        if i_ave == 0:
            variables[name] = var / window_ave
        else:
            variables[name] += var / window_ave

with open(os.path.join(case_path, "qoi.pkl"), "wb") as f:
    pickle.dump(variables, f)
