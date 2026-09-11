import os

import numpy as np
import pandas as pd

# the 4 QoI value columns the surrogate can target (each has a matching _err)
QOI_VALUE_COLS = ["qoi", "qoi_kla", "qoi_sum", "qoi_sum_kla"]
QOI_ERR_COLS = ["qoi_err", "qoi_kla_err", "qoi_sum_err", "qoi_sum_kla_err"]

# QoI assigned to unreliable designs, and the SNR gate for "reliable"
FAILED_QOI_VALUE = 0.0
SNR_THRESHOLD = 5.0


def _is_reliable(value: float, err: float, threshold: float) -> bool:
    """True when the signal-to-noise ratio |value| / err meets the threshold."""
    return err > 0 and abs(value) / err >= threshold


def load_full_dataset(study_path: str) -> tuple[np.ndarray, pd.DataFrame]:
    converged_X = pd.read_csv(
        os.path.join(study_path, "X_data.csv"), header=None
    ).values
    converged_Y = pd.read_csv(os.path.join(study_path, "Y_data.csv"))

    X = converged_X.astype(int)
    Y_rows = []
    for row_index in range(len(converged_Y)):
        source = converged_Y.iloc[row_index]
        row = {"sim_id": int(source["sim_id"])}
        for qoi, err_col in zip(QOI_VALUE_COLS, QOI_ERR_COLS):
            value = float(source[qoi])
            err = float(source[err_col])
            row[qoi] = (
                value
                if _is_reliable(value, err, SNR_THRESHOLD)
                else FAILED_QOI_VALUE
            )
            row[err_col] = err
        Y_rows.append(row)

    column_order = ["sim_id"] + QOI_VALUE_COLS + QOI_ERR_COLS
    Y = pd.DataFrame(Y_rows)[column_order].reset_index(drop=True)
    Y["sim_id"] = Y["sim_id"].astype(int)
    return X, Y
