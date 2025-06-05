import argparse
import os
import sys

import numpy as np
from prettyPlot.plotting import plt, pretty_labels

from bird.utilities.ofio import *


def compute_gas_holdup(caseFolder, timeFolder, nCells, field_dict={}):
    if not ("alpha_liq" in field_dict) or field_dict["alpha_liq"] is None:
        alpha_liq_file = os.path.join(caseFolder, timeFolder, "alpha.liquid")
        alpha_liq = readOFScal(alpha_liq_file, nCells)
        # print("reading alpha_liq")
        field_dict["alpha_liq"] = alpha_liq
    if not ("volume" in field_dict) or field_dict["volume"] is None:
        volume_file = os.path.join(caseFolder, "0", "V")
        volume = readOFScal(volume_file, nCells)
        # print("reading Volume")
        field_dict["volume"] = volume
    alpha_liq = field_dict["alpha_liq"]
    volume = field_dict["volume"]
    holdup = np.sum((1 - alpha_liq) * volume) / np.sum(volume)
    return holdup, field_dict


def co2liq(caseFolder, timeFolder, nCells, field_dict={}):
    if not ("alpha_liq" in field_dict) or field_dict["alpha_liq"] is None:
        alpha_liq_file = os.path.join(caseFolder, timeFolder, "alpha.liquid")
        alpha_liq = readOFScal(alpha_liq_file, nCells)
        # print("reading alpha_liq")
        field_dict["alpha_liq"] = alpha_liq
    if not ("co2_liq" in field_dict) or field_dict["co2_liq"] is None:
        co2_liq_file = os.path.join(caseFolder, timeFolder, "CO2.liquid")
        co2_liq = readOFScal(co2_liq_file, nCells)
        # print("computing co2 liq")
        field_dict["co2_liq"] = co2_liq
    if not ("volume" in field_dict) or field_dict["volume"] is None:
        volume_file = os.path.join(caseFolder, "0", "V")
        volume = readOFScal(volume_file, nCells)
        # print("reading Volume")
        field_dict["volume"] = volume
    if not ("indliq" in field_dict) or field_dict["indliq"] is None:
        alpha_liq = field_dict["alpha_liq"]
        indliq = np.argwhere(alpha_liq > 0.5)
        # print("computing indliq")
        field_dict["indliq"] = indliq
    volume = field_dict["volume"]
    indliq = field_dict["indliq"]
    alpha_liq = field_dict["alpha_liq"]
    co2_liq = field_dict["co2_liq"]
    met = np.sum(
        alpha_liq[indliq] * co2_liq[indliq] * volume[indliq]
    ) / np.sum(volume[indliq])
    return met, field_dict


def cliq(caseFolder, timeFolder, nCells, field_dict={}):
    if not ("alpha_liq" in field_dict) or field_dict["alpha_liq"] is None:
        alpha_liq_file = os.path.join(caseFolder, timeFolder, "alpha.liquid")
        alpha_liq = readOFScal(alpha_liq_file, nCells)
        # print("reading alpha_liq")
        field_dict["alpha_liq"] = alpha_liq
    if not ("rho_liq" in field_dict) or field_dict["rho_liq"] is None:
        rho_liq_file = os.path.join(caseFolder, timeFolder, "rhom")
        rho_liq = readOFScal(rho_liq_file, nCells)
        field_dict["rho_liq"] = rho_liq
    if not ("co2_liq" in field_dict) or field_dict["co2_liq"] is None:
        co2_liq_file = os.path.join(caseFolder, timeFolder, "CO2.liquid")
        co2_liq = readOFScal(co2_liq_file, nCells)
        # print("computing co2 liq")
        field_dict["co2_liq"] = co2_liq
    if not ("h2_liq" in field_dict) or field_dict["h2_liq"] is None:
        h2_liq_file = os.path.join(caseFolder, timeFolder, "H2.liquid")
        h2_liq = readOFScal(h2_liq_file, nCells)
        # print("computing h2 liq")
        field_dict["h2_liq"] = h2_liq
    if not ("volume" in field_dict) or field_dict["volume"] is None:
        volume_file = os.path.join(caseFolder, "0", "V")
        volume = readOFScal(volume_file, nCells)
        # print("reading Volume")
        field_dict["volume"] = volume
    if not ("indliq" in field_dict) or field_dict["indliq"] is None:
        alpha_liq = field_dict["alpha_liq"]
        indliq = np.argwhere(alpha_liq > 0.5)
        # print("computing indliq")
        field_dict["indliq"] = indliq

    volume = field_dict["volume"]
    indliq = field_dict["indliq"]
    alpha_liq = field_dict["alpha_liq"]
    co2_liq = field_dict["co2_liq"]
    h2_liq = field_dict["h2_liq"]
    rho_liq = field_dict["rho_liq"]

    # c_h2 = rho_liq[indliq] * alpha_liq[indliq] * h2_liq[indliq] / 0.002016
    # c_co2 = rho_liq[indliq] * alpha_liq[indliq] * co2_liq[indliq] / 0.04401

    c_h2 = 1000 * h2_liq[indliq] / 0.002016
    c_co2 = 1000 * co2_liq[indliq] / 0.04401

    c_h2 = np.sum(c_h2 * volume[indliq] * alpha_liq[indliq]) / np.sum(
        volume[indliq] * alpha_liq[indliq]
    )
    c_co2 = np.sum(c_co2 * volume[indliq] * alpha_liq[indliq]) / np.sum(
        volume[indliq] * alpha_liq[indliq]
    )

    return c_co2, c_h2, field_dict


def h2liq(caseFolder, timeFolder, nCells, field_dict={}):
    if not ("alpha_liq" in field_dict) or field_dict["alpha_liq"] is None:
        alpha_liq_file = os.path.join(caseFolder, timeFolder, "alpha.liquid")
        alpha_liq = readOFScal(alpha_liq_file, nCells)
        # print("reading alpha_liq")
        field_dict["alpha_liq"] = alpha_liq
    if not ("h2_liq" in field_dict) or field_dict["h2_liq"] is None:
        h2_liq_file = os.path.join(caseFolder, timeFolder, "H2.liquid")
        h2_liq = readOFScal(h2_liq_file, nCells)
        # print("computing h2 liq")
        field_dict["h2_liq"] = h2_liq
    if not ("volume" in field_dict) or field_dict["volume"] is None:
        volume_file = os.path.join(caseFolder, "0", "V")
        volume = readOFScal(volume_file, nCells)
        # print("reading Volume")
        field_dict["volume"] = volume
    if not ("indliq" in field_dict) or field_dict["indliq"] is None:
        alpha_liq = field_dict["alpha_liq"]
        indliq = np.argwhere(alpha_liq > 0.5)
        # print("computing indliq")
        field_dict["indliq"] = indliq
    volume = field_dict["volume"]
    indliq = field_dict["indliq"]
    alpha_liq = field_dict["alpha_liq"]
    h2_liq = field_dict["h2_liq"]
    met = np.sum(alpha_liq[indliq] * h2_liq[indliq] * volume[indliq]) / np.sum(
        volume[indliq]
    )
    return met, field_dict


def vol_liq(caseFolder, timeFolder, nCells, field_dict={}):
    if not ("alpha_liq" in field_dict) or field_dict["alpha_liq"] is None:
        alpha_liq_file = os.path.join(caseFolder, timeFolder, "alpha.liquid")
        alpha_liq = readOFScal(alpha_liq_file, nCells)
        # print("reading alpha_liq")
        field_dict["alpha_liq"] = alpha_liq
    if not ("volume" in field_dict) or field_dict["volume"] is None:
        volume_file = os.path.join(caseFolder, "0", "V")
        volume = readOFScal(volume_file, nCells)
        # print("reading Volume")
        field_dict["volume"] = volume
    volume = field_dict["volume"]
    alpha_liq = field_dict["alpha_liq"]
    indliq = np.argwhere(alpha_liq > 0.0)
    liqvol = np.sum(alpha_liq[indliq] * volume[indliq]) / np.sum(
        volume[indliq]
    )
    return liqvol, field_dict


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

if os.path.isfile(os.path.join(dataFolder, case_name, "conv.npz")):
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
        if "volume" in field_dict:
            new_field_dict["volume"] = field_dict["volume"]
        field_dict = new_field_dict
    gh_history[itime], field_dict = compute_gas_holdup(
        case_path, time_str_sorted[itime], nCells, field_dict
    )
    co2_history[itime], field_dict = co2liq(
        case_path, time_str_sorted[itime], nCells, field_dict
    )
    h2_history[itime], field_dict = h2liq(
        case_path, time_str_sorted[itime], nCells, field_dict
    )
    liqvol_history[itime], field_dict = vol_liq(
        case_path, time_str_sorted[itime], nCells, field_dict
    )
    c_co2_history[itime], c_h2_history[itime], field_dict = cliq(
        case_path, time_str_sorted[itime], nCells, field_dict
    )


os.makedirs(dataFolder, exist_ok=True)
os.makedirs(os.path.join(dataFolder, case_name), exist_ok=True)
np.savez(
    os.path.join(dataFolder, case_name, "conv.npz"),
    time=np.array(time_float_sorted),
    gh=gh_history,
    co2=co2_history,
    h2=h2_history,
    vol_liq=liqvol_history,
    c_h2=c_h2_history,
    c_co2=c_co2_history,
)
