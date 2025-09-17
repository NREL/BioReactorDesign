import os
import shutil
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

    # Make sure None arguments are correctly handled
    n_cells = len(field_dict["alpha.liquid"])
    time_folder = "79"
    gh1, _ = compute_gas_holdup(
        case_folder=case_folder, time_folder=time_folder
    )
    gh2, _ = compute_gas_holdup(
        case_folder=case_folder, n_cells=n_cells, time_folder=time_folder
    )

    # Results need to be exactly the same
    assert abs(gh1 - gh) < 1e-12
    assert abs(gh2 - gh) < 1e-12


def test_compute_diam():
    """
    Test for bubble diameter calculation
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
    diam, field_dict = compute_ave_bubble_diam(
        time_folder="1", field_dict=field_dict, **kwargs
    )
    field_dict = {}
    diam, field_dict = compute_ave_bubble_diam(
        time_folder="79", field_dict=field_dict, **kwargs
    )

    # Make sure None arguments are correctly handled
    n_cells = len(field_dict["d.gas"])
    time_folder = "79"
    diam1, _ = compute_ave_bubble_diam(
        case_folder=case_folder, time_folder=time_folder
    )
    diam2, _ = compute_ave_bubble_diam(
        case_folder=case_folder, n_cells=n_cells, time_folder=time_folder
    )

    # Results need to be exactly the same
    assert abs(diam1 - diam) < 1e-12
    assert abs(diam2 - diam) < 1e-12


def test_compute_superficial_gas_velocity():
    """
    Test for superficial gas velocity calculation
    """
    case_folder = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "bird",
        "postprocess",
        "data_conditional_mean/",
    )
    kwargs = {
        "case_folder": case_folder,
        "n_cells": None,
        "volume_time": "1",
        "direction": 1,
        "cell_centers_file": "meshCellCentres_1.obj",
    }
    field_dict = {}
    sup_vel, field_dict = compute_superficial_gas_velocity(
        time_folder="79", field_dict=field_dict, **kwargs
    )

    # Make sure None arguments are correctly handled
    n_cells = len(field_dict["V"])
    time_folder = "79"
    sup_vel1, _ = compute_superficial_gas_velocity(
        case_folder=case_folder, time_folder=time_folder, direction=1
    )
    sup_vel2, _ = compute_superficial_gas_velocity(
        case_folder=case_folder,
        n_cells=n_cells,
        time_folder=time_folder,
        direction=1,
    )

    # Results need to be exactly the same
    assert abs(sup_vel1 - sup_vel) < 1e-12
    assert abs(sup_vel2 - sup_vel) < 1e-12

    # Do the calculation with paraview
    sup_vel3, _ = compute_superficial_gas_velocity(
        case_folder=case_folder,
        time_folder=time_folder,
        direction=1,
        use_pv=True,
    )
    # Make sure different methods agree with less than 1% error
    assert abs((sup_vel3 - sup_vel2) / sup_vel2) < 0.01

    # Make sure that we don't use paraview if not possible
    polyMesh_dir = os.path.join(case_folder, "constant", "polyMesh")
    shutil.move(os.path.join(polyMesh_dir, "faces"), ".")
    sup_vel4, _ = compute_superficial_gas_velocity(
        case_folder=case_folder,
        time_folder=time_folder,
        direction=1,
        use_pv=True,
    )
    # Results need to be exactly the same
    assert abs(sup_vel4 - sup_vel) < 1e-12
    shutil.move("faces", polyMesh_dir)


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

    # Make sure None arguments are correctly handled
    n_cells = len(field_dict["H2.liquid"])
    time_folder = kwargs["time_folder"]
    ave_y_h21, _ = compute_ave_y_liq(
        case_folder=case_folder, time_folder=time_folder, spec_name="H2"
    )
    ave_y_h22, _ = compute_ave_y_liq(
        case_folder=case_folder,
        n_cells=n_cells,
        time_folder=time_folder,
        spec_name="H2",
    )

    # Results need to be exactly the same
    assert abs(ave_y_h21 - ave_y_h2) < 1e-12
    assert abs(ave_y_h22 - ave_y_h2) < 1e-12


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
    # Make sure None arguments are correctly handled
    n_cells = len(field_dict["H2.liquid"])
    time_folder = kwargs["time_folder"]
    ave_conc_h21, _ = compute_ave_conc_liq(
        case_folder=case_folder,
        time_folder=time_folder,
        spec_name="H2",
        mol_weight=2.01594 * 1e-3,
    )
    ave_conc_h22, _ = compute_ave_conc_liq(
        case_folder=case_folder,
        time_folder=time_folder,
        spec_name="H2",
        mol_weight=2.01594 * 1e-3,
        n_cells=n_cells,
    )

    # Results need to be exactly the same
    assert abs(ave_conc_h21 - ave_conc_h2) < 1e-12
    assert abs(ave_conc_h22 - ave_conc_h2) < 1e-12


if __name__ == "__main__":
    test_compute_superficial_gas_velocity()
    test_compute_gh()
    # test_ave_y_liq()
    # test_ave_conc_liq()
