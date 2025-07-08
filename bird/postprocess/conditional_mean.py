import os
import pickle

from prettyPlot.plotting import plt

from bird.utilities.bubble_col_util import *
from bird.utilities.mathtools import *
from bird.utilities.ofio import *


def compute_cond_mean(
    case_path,
    vert_ind,
    field_name_list,
    window_ave,
    n_bins=32,
    diff_val_list=[],
    diff_name_list=[],
):
    time_float_sorted, time_str_sorted = getCaseTimes(case_path)
    mesh_time_str = getMeshTime(case_path)
    cellCentres = readMesh(
        os.path.join(case_path, f"meshCellCentres_{mesh_time_str}.obj")
    )
    nCells = len(cellCentres)
    assert len(diff_val_list) == len(diff_name_list)
    window_ave = min(window_ave, len(time_str_sorted))

    fields_cond = {}
    fields_cond_tmp = {}
    for name in field_name_list:
        fields_cond[name] = {}
        fields_cond_tmp[name] = {}

    print(f"Case : {case_path}")

    for i_ave in range(window_ave):
        time_folder = time_str_sorted[-i_ave - 1]
        print(f"\tReading Time : {time_folder}")
        field_file = []
        for field_name in field_name_list:
            field_file.append(os.path.join(case_path, time_folder, field_name))

        # if os.path.isfile(d_gas_file):
        #    has_d = True
        # else:
        #    has_d = False

        for filename, name in zip(field_file, field_name_list):
            val_dict = {}
            if name.lower() == "kla_co":
                if "D_CO" in diff_name_list:
                    diff = diff_val_list[diff_name_list.index("D_CO")]
                else:
                    diff = None
                field_tmp, val_dict = computeSpec_kla_field(
                    os.path.join(case_path, time_folder),
                    nCells,
                    key_suffix="co",
                    cellCentres=cellCentres,
                    val_dict=val_dict,
                    diff=diff,
                )
            elif name.lower() == "kla_co2":
                if "D_CO2" in diff_name_list:
                    diff = diff_val_list[diff_name_list.index("D_CO2")]
                else:
                    diff = None
                field_tmp, val_dict = computeSpec_kla_field(
                    os.path.join(case_path, time_folder),
                    nCells,
                    key_suffix="co2",
                    cellCentres=cellCentres,
                    val_dict=val_dict,
                    diff=diff,
                )
            elif name.lower() == "kla_h2":
                if "D_H2" in diff_name_list:
                    diff = diff_val_list[diff_name_list.index("D_H2")]
                else:
                    diff = None
                field_tmp, val_dict = computeSpec_kla_field(
                    os.path.join(case_path, time_folder),
                    nCells,
                    key_suffix="h2",
                    cellCentres=cellCentres,
                    val_dict=val_dict,
                    diff=diff,
                )
            else:
                field_tmp = readOFScal(filename, nCells)["field"]
            vert_axis, field_cond_tmp = conditionalAverage(
                cellCentres[:, vert_ind], field_tmp, nbin=n_bins
            )
            if i_ave == 0:
                fields_cond[name]["val"] = field_cond_tmp / window_ave
                fields_cond[name]["vert"] = vert_axis
            else:
                fields_cond[name]["val"] += field_cond_tmp / window_ave

    return fields_cond


def save_cond(filename, fields_cond):
    with open(filename, "wb") as f:
        pickle.dump(fields_cond, f)


def sequencePlot(
    cond,
    folder_names,
    field_name,
    case_names=[],
    symbList=["-", "-d", "-^", "-.", "-s", "-o", "-+"],
):
    if not len(case_names) == len(folder_names):
        case_names = [f"test{i}" for i in range(len(folder_names))]
    if len(case_names) > len(symbList):
        print(
            f"ERROR: too many cases ({len(case_names)}), reduce number of case to {len(symbList)} or add symbols"
        )
        sys.exit()
    for ic, (case_name, folder_name) in enumerate(
        zip(case_names, folder_names)
    ):
        label = ""
        if ic == 0:
            label = case_names[ic]
        plt.plot(
            cond[folder_name][field_name]["val"],
            cond[folder_name][field_name]["vert"],
            symbList[ic],
            markersize=10,
            markevery=10,
            linewidth=3,
            color="k",
            label=label,
        )
    if field_name == "alpha.gas":
        plot_name = "gasHoldup"
    else:
        plot_name = field_name
    return plot_name
