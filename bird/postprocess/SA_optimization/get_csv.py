import csv
import os

import numpy as np
import pickle5 as pkl


def get_config_result(config_fold):
    with open(os.path.join(config_fold, "results.pkl"), "rb") as f:
        results = pkl.load(f)
    with open(os.path.join(config_fold, "configs.pkl"), "rb") as f:
        configs = pkl.load(f)
    Xdata = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], np.int64)
    count = 0
    # Save data into CSV files
    xfname = "Xdata_" + config_fold + ".csv"
    yfname = "ydata_" + config_fold + ".csv"
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


studies = {"study_scaleup_0_4vvm_3000W": r"0.0036$m^3$ 0.4vvm 0W"}


for study in studies:
    get_config_result(study)
