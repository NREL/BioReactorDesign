import os
from pathlib import Path

from prettyPlot.plotting import plt, pretty_labels

from bird.postprocess.post_quantities import *


def test_compute_gh():
    """
    Test for gas holdup calculation
    """
    case_folder = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "bird",
        "postprocess",
        "data_conditional_mean",
    )
    kwargs = {"case_folder": case_folder, "n_cells": None, "volume_time": "1"}
    field_dict = {}
    gh, field_dict = compute_gas_holdup(
        time_folder="1", field_dict=field_dict, **kwargs
    )
    field_dict = {}
    gh, field_dict = compute_gas_holdup(
        time_folder="79", field_dict=field_dict, **kwargs
    )


def test_ave_y_liq():
    """
    Test for liquid volume averaged species mass fraction
    """
    case_folder = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "bird",
        "postprocess",
        "data_conditional_mean",
    )
    kwargs = {
        "time_folder": "79",
        "case_folder": case_folder,
        "n_cells": None,
        "volume_time": "1",
    }
    field_dict = {}
    ave_y_co2, field_dict = compute_ave_y_liq(
        spec_name="CO2", field_dict=field_dict, **kwargs
    )
    ave_y_co, field_dict = compute_ave_y_liq(
        spec_name="CO", field_dict=field_dict, **kwargs
    )
    ave_y_h2, field_dict = compute_ave_y_liq(
        spec_name="H2", field_dict=field_dict, **kwargs
    )


def test_ave_conc_liq():
    """
    Test for liquid volume averaged species concentration
    """
    case_folder = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "bird",
        "postprocess",
        "data_conditional_mean",
    )
    kwargs = {
        "time_folder": "79",
        "case_folder": case_folder,
        "n_cells": None,
        "volume_time": "1",
    }
    field_dict = {}
    ave_conc_co2, field_dict = compute_ave_conc_liq(
        spec_name="CO2",
        mol_weight=44.00995 * 1e-3,
        field_dict=field_dict,
        **kwargs,
    )
    ave_conc_co, field_dict = compute_ave_conc_liq(
        spec_name="CO",
        mol_weight=28.01055 * 1e-3,
        field_dict=field_dict,
        **kwargs,
    )
    ave_conc_h2, field_dict = compute_ave_conc_liq(
        spec_name="H2",
        mol_weight=2.01594 * 1e-3,
        field_dict=field_dict,
        **kwargs,
    )


if __name__ == "__main__":
    test_compute_gh()
    test_ave_y_liq()
    test_ave_conc_liq()
