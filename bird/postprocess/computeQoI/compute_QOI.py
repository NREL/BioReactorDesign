import argparse
import os
import pickle
import sys

import numpy as np

from bird.postprocess.post_quantities import *
from bird.utilities.ofio import *

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
        "gh",
        "d",
        "CO2_liq",
        "CO_liq",
        "H2_liq",
        "c_CO2_liq",
        "c_CO_liq",
        "c_H2_liq",
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
parser.add_argument(
    "-conv",
    "--windowConv",
    type=int,
    metavar="",
    required=False,
    help="Window Convergence",
    default=1,
)
parser.add_argument(
    "-dv",
    "--diff_val_list",
    nargs="+",
    type=float,
    help="List of diffusivities",
    default=[],
    required=False,
)
parser.add_argument(
    "-dn",
    "--diff_name_list",
    nargs="+",
    type=str,
    help="List of diffusivities",
    default=[],
    required=False,
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
diff_val_list = args.diff_val_list
diff_name_list = args.diff_name_list
assert len(diff_val_list) == len(diff_name_list)

window_ave = min(args.windowAve, len(time_str_sorted) - 1)
window_conv = min(args.windowConv, len(time_str_sorted) - 1)

variables = {}
variables_conv = {}
for name in var_name_list:
    variables_conv[name] = {}
    variables_conv[name]["x"] = []
    variables_conv[name]["y"] = []


def get_var(
    case_path, time_folder, mesh_time_str, cellCentres, nCells, val_dict, name
):
    localFolder = os.path.join(case_path, time_folder)
    localFolder_vol = os.path.join(case_path, mesh_time_str)
    if name.lower() == "gh":
        var, val_dict = compute_gas_holdup(
            case_path,
            time_folder,
            nCells,
            volume_time="0",
            field_dict=val_dict,
        )
    elif name.lower() == "d":
        var, val_dict = compute_ave_bubble_diam(
            case_path,
            time_folder,
            nCells,
            volume_time="0",
            field_dict=val_dict,
        )
    elif name.lower() == "co2_liq":
        var, val_dict = compute_ave_y_liq(
            case_path,
            time_folder,
            nCells,
            volume_time="0",
            spec_name="CO2",
            field_dict=val_dict,
        )
    elif name.lower() == "co_liq":
        var, val_dict = compute_ave_y_liq(
            case_path,
            time_folder,
            nCells,
            volume_time="0",
            spec_name="CO",
            field_dict=val_dict,
        )
    elif name.lower() == "h2_liq":
        var, val_dict = compute_ave_y_liq(
            case_path,
            time_folder,
            nCells,
            volume_time="0",
            spec_name="H2",
            field_dict=val_dict,
        )
    elif name.lower() == "c_co2_liq":
        var, val_dict = compute_ave_conc_liq(
            case_path,
            time_folder,
            nCells,
            volume_time="0",
            spec_name="CO2",
            mol_weight=44.00995 * 1e-3,
            field_dict=val_dict,
        )
    elif name.lower() == "c_co_liq":
        var, val_dict = compute_ave_conc_liq(
            case_path,
            time_folder,
            nCells,
            volume_time="0",
            spec_name="CO",
            mol_weight=28.01055 * 1e-3,
            field_dict=val_dict,
        )
    elif name.lower() == "c_h2_liq":
        var, val_dict = compute_ave_conc_liq(
            case_path,
            time_folder,
            nCells,
            volume_time="0",
            spec_name="H2",
            mol_weight=2.01594 * 1e-3,
            field_dict=val_dict,
        )
    else:
        raise NotImplementedError(f"Unknown variable {name}")

    return var, val_dict


print(f"Case : {case_path}")

print(f"Window Ave")
for i_ave in range(window_ave):
    time_folder = time_str_sorted[-i_ave - 1]
    print(f"\tTime : {time_folder}")
    case_variables = []
    val_dict = {}
    for name in var_name_list:
        var, val_dict = get_var(
            case_path,
            time_folder,
            mesh_time_str,
            cellCentres,
            nCells,
            val_dict=val_dict,
            name=name,
        )

        if i_ave == 0:
            variables[name] = var / window_ave
        else:
            variables[name] += var / window_ave


print(f"Window Conv")
for i_conv in range(window_conv):
    time_folder = time_str_sorted[-window_conv + i_conv]
    print(f"\tTime : {time_folder}")
    case_variables = []
    val_dict = {}
    for name in var_name_list:
        var, val_dict = get_var(
            case_path,
            time_folder,
            mesh_time_str,
            cellCentres,
            nCells,
            val_dict=val_dict,
            name=name,
        )
        variables_conv[name]["x"] += [float(time_folder)]
        variables_conv[name]["y"] += [var]


with open(os.path.join(case_path, "qoi.pkl"), "wb") as f:
    pickle.dump(variables, f)
with open(os.path.join(case_path, "qoi_conv.pkl"), "wb") as f:
    pickle.dump(variables_conv, f)
