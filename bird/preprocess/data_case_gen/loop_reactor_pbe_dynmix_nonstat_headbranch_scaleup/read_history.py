import argparse
import os
import sys

import numpy as np
from prettyPlot.plotting import plt, pretty_labels

from bird.postprocess.post_quantities import *
from bird.utilities.ofio import *

parser = argparse.ArgumentParser(description="Convergence of GH")
parser.add_argument(
    "-cn",
    "--case_name",
    type=str,
    metavar="",
    required=True,
    help="Case name",
)
parser.add_argument(
    "-df",
    "--data_folder",
    type=str,
    metavar="",
    required=False,
    help="data folder name",
    default="data",
)

args, unknown = parser.parse_known_args()


case_root = "."  # "../"
case_name = args.case_name  # "12_hole_sparger_snappyRefine_700rpm_opt_coeff"
case_path = "."
dataFolder = args.data_folder

if os.path.isfile(os.path.join(dataFolder, case_name, "conv2.npz")):
    sys.exit("WARNING: History already created, Skipping")

time_float_sorted, time_str_sorted = getCaseTimes(case_path, remove_zero=True)
cellCentres = readMesh(os.path.join(case_path, f"meshCellCentres_0.obj"))
nCells = len(cellCentres)


co2_history = np.zeros(len(time_str_sorted))
c_co2_history = np.zeros(len(time_str_sorted))
h2_history = np.zeros(len(time_str_sorted))
c_h2_history = np.zeros(len(time_str_sorted))
gh_history = np.zeros(len(time_str_sorted))
liqvol_history = np.zeros(len(time_str_sorted))
print(f"case_path = {case_path}")
field_dict = {}
for itime, time in enumerate(time_float_sorted):
    time_folder = time_str_sorted[itime]
    print(f"\tTime : {time_folder}")
    if not field_dict == {}:
        new_field_dict = {}
        if "V" in field_dict:
            new_field_dict["V"] = field_dict["V"]
        field_dict = new_field_dict
    gh_history[itime], field_dict = compute_gas_holdup(
        case_path,
        time_str_sorted[itime],
        nCells,
        volume_time="0",
        field_dict=field_dict,
    )
    co2_history[itime], field_dict = compute_ave_y_liq(
        case_path,
        time_str_sorted[itime],
        nCells,
        volume_time="0",
        spec_name="CO2",
        field_dict=field_dict,
    )
    h2_history[itime], field_dict = compute_ave_y_liq(
        case_path,
        time_str_sorted[itime],
        nCells,
        volume_time="0",
        spec_name="H2",
        field_dict=field_dict,
    )
    c_co2_history[itime], field_dict = compute_ave_conc_liq(
        case_path,
        time_str_sorted[itime],
        nCells,
        volume_time="0",
        spec_name="CO2",
        mol_weight=0.04401,
        field_dict=field_dict,
    )
    c_h2_history[itime], field_dict = compute_ave_conc_liq(
        case_path,
        time_str_sorted[itime],
        nCells,
        volume_time="0",
        spec_name="H2",
        mol_weight=0.002016,
        field_dict=field_dict,
    )

os.makedirs(dataFolder, exist_ok=True)
os.makedirs(os.path.join(dataFolder, case_name), exist_ok=True)
np.savez(
    os.path.join(dataFolder, case_name, "conv2.npz"),
    time=np.array(time_float_sorted),
    gh=gh_history,
    co2=co2_history,
    h2=h2_history,
    c_h2=c_h2_history,
    c_co2=c_co2_history,
)
