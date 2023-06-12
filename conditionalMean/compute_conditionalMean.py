import argparse
import sys

import numpy as np

sys.path.append("util")
import os
import pickle

from ofio import *

parser = argparse.ArgumentParser(description="Case folder")
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
    "-vert",
    "--verticalDirection",
    type=int,
    metavar="",
    required=False,
    help="Index of vertical direction",
    default=1,
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
    "-fl",
    "--field_list",
    nargs="+",
    help="List of fields to plot",
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
    required=False,
)
args = parser.parse_args()


case_path = args.caseFolder
vert_ind = args.verticalDirection
field_name_list = args.field_list
time_float_sorted, time_str_sorted = getCaseTimes(case_path)
mesh_time_str = getMeshTime(case_path)
cellCentres = readMesh(
    os.path.join(case_path, f"meshCellCentres_{mesh_time_str}.obj")
)
nCells = len(cellCentres)

window_ave = min(args.windowAve, len(time_str_sorted))

nbins = 32

fields_cond = {}
fields_cond_tmp = {}
for name in field_name_list:
    fields_cond[name] = {}
    fields_cond_tmp[name] = {}


print(f"Case : {case_path}")


for i_ave in range(window_ave):
    time_folder = time_str_sorted[-i_ave - 1]
    print(f"\tTime : {time_folder}")
    field_file = []
    for field_name in field_name_list:
        field_file.append(os.path.join(case_path, time_folder, field_name))

    # if os.path.isfile(d_gas_file):
    #    has_d = True
    # else:
    #    has_d = False

    for filename, name in zip(field_file, field_name_list):
        field_tmp = readOFScal(filename, nCells)
        vert_axis, field_cond_tmp = conditionalAverage(
            cellCentres[:, vert_ind], field_tmp, nbin=nbins
        )
        if i_ave == 0:
            fields_cond[name]["val"] = field_cond_tmp / window_ave
            fields_cond[name]["vert"] = vert_axis
        else:
            fields_cond[name]["val"] += field_cond_tmp / window_ave

    # if has_d:
    #    d_gas = readOFScal(d_gas_file,nCells)
    #    y_axis, d_cond_tmp = conditionalAverage(cellCentres[:,1], d_gas, nbin=nbins)
    # else:
    #    d_cond_tmp = np.ones(nbins)*2.86e-3

with open(os.path.join(case_path, "cond.pkl"), "wb") as f:
    pickle.dump(fields_cond, f)
