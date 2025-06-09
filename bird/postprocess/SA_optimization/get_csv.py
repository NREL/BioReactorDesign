import csv
import os
import pickle as pkl

import numpy as np


def get_config_result(study_fold: str = ".") -> None:
    """
    Read the configs.pkl and results.pkl files from a study
    Saves the configuration in Xdata_{study_fold}.csv file
    Save the qoi and qoi_error in ydata_{study_fold}.csv file

    Parameters
    ----------
    study_fold : str
        Folder that contains the study results

    Returns
    ----------
    None

    """
    # Read results
    with open(os.path.join(study_fold, "results.pkl"), "rb") as f:
        results = pkl.load(f)
    with open(os.path.join(study_fold, "configs.pkl"), "rb") as f:
        configs = pkl.load(f)

    Xdata = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], np.int64)
    count = 0

    # Save data into CSV files
    xfname = os.path.join(study_fold, f"Xdata.csv")
    yfname = os.path.join(study_fold, f"ydata.csv")
    with open(xfname, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for sims in results:
            b0 = configs[sims][0]
            b1 = configs[sims][1]
            b2 = configs[sims][2]
            raw_data = np.concatenate((b0, b1, b2), axis=None)
            writer.writerow(raw_data)

    with open(yfname, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for sims in results:
            q0 = results[sims]["qoi"]
            q1 = results[sims]["qoi_err"]
            y_data = np.concatenate((q0, q1), axis=None)
            writer.writerow(y_data)


if __name__ == "__main__":
    studies = {
        "study_scaleup_0_4vvm_3000W": r"608$m^3$ 0.4vvm 3000W",
        "study_scaleup_0_1vvm_6000W": r"608$m^3$ 0.1vvm 6000W",
        "study_0_4vvm_1W": r"0.00361$m^3$ 0.4vvm 1W",
    }
    for study in studies:
        get_config_result(study)
