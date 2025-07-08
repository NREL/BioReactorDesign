import os
from pathlib import Path

from bird.postprocess.early_pred import (
    bayes_fit,
    fit_and_ext,
    multi_data_load,
    plotAllEarly,
    plotAllEarly_uq,
)


def test_fit():
    BIRD_EARLY_PRED_DATA_DIR = os.path.join(
        Path(__file__).parent, "..", "..", "bird", "postprocess", "data_early"
    )
    data_dict, color_files = multi_data_load(BIRD_EARLY_PRED_DATA_DIR)
    data_dict = fit_and_ext(data_dict)
    plotAllEarly(data_dict, color_files=color_files, chop=True, extrap=True)


def test_bayes_fit():
    BIRD_EARLY_PRED_DATA_DIR = os.path.join(
        Path(__file__).parent, "..", "..", "bird", "postprocess", "data_early"
    )
    data_dict, color_files = multi_data_load(BIRD_EARLY_PRED_DATA_DIR)
    bayes_fit(data_dict)
    plotAllEarly_uq(data_dict, color_files=color_files)
