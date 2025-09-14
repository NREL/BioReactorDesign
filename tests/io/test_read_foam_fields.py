import os
from pathlib import Path

import numpy as np

from bird.utilities.ofio import readOF, readOFScal, readOFVec


def test_read_nonunif_scal():
    """
    Test for reading non uniform scalarField
    """
    case_folder = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "bird",
        "postprocess",
        "data_conditional_mean",
    )
    # Read non uniform field
    data_dict = readOFScal(filename=os.path.join(case_folder, "79", "CO2.gas"))
    assert abs(data_dict["field"][0] - 0.616955) < 1e-6
    assert abs(data_dict["field"][-1] - 0.625389) < 1e-6
    assert abs(data_dict["n_cells"] - 137980) < 1e-6
    assert abs(data_dict["field"].shape[0] - 137980) < 1e-6
    assert data_dict["name"] == "CO2.gas"
    # Read non uniform field with flexible interface
    data_dict = readOF(filename=os.path.join(case_folder, "79", "CO2.gas"))
    assert abs(data_dict["field"][0] - 0.616955) < 1e-6
    assert abs(data_dict["field"][-1] - 0.625389) < 1e-6
    assert abs(data_dict["n_cells"] - 137980) < 1e-6
    assert abs(data_dict["field"].shape[0] - 137980) < 1e-6
    assert data_dict["name"] == "CO2.gas"


def test_read_unif_scal():
    """
    Test for reading uniform scalarField
    """
    case_folder = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "bird",
        "postprocess",
        "data_conditional_mean",
    )
    # Read non uniform field
    data_dict = readOFScal(filename=os.path.join(case_folder, "79", "f.gas"))
    assert abs(data_dict["field"] - 1) < 1e-6
    assert data_dict["n_cells"] is None
    # Read non uniform field with flexible interface
    data_dict = readOF(filename=os.path.join(case_folder, "79", "f.gas"))
    assert abs(data_dict["field"] - 1) < 1e-6
    assert data_dict["n_cells"] is None
    # Read non uniform field with prespecified cell number
    data_dict = readOFScal(
        filename=os.path.join(case_folder, "79", "f.gas"), n_cells=100
    )
    assert np.shape(data_dict["field"]) == (100,)
    assert np.linalg.norm(data_dict["field"] - 1) < 1e-6
    assert data_dict["n_cells"] == 100
    assert data_dict["name"] == "f.gas"


def test_read_nonunif_vec():
    """
    Test for reading non uniform vectorField
    """
    case_folder = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "bird",
        "postprocess",
        "data_conditional_mean",
    )
    # Read non uniform field
    data_dict = readOFVec(filename=os.path.join(case_folder, "79", "U.gas"))
    assert (
        np.linalg.norm(
            data_dict["field"][0, :] - [0.140018, 1.20333, 0.127566]
        )
        < 1e-6
    )
    assert (
        np.linalg.norm(
            data_dict["field"][-1, :] - [0.0409271, 0.176052, 0.0302899]
        )
        < 1e-6
    )
    assert abs(data_dict["n_cells"] - 137980) < 1e-6
    assert abs(data_dict["field"].shape[0] - 137980) < 1e-6
    assert data_dict["name"] == "U.gas"
    # Read non uniform field with flexible interface
    data_dict = readOF(filename=os.path.join(case_folder, "79", "U.gas"))
    assert (
        np.linalg.norm(
            data_dict["field"][0, :] - [0.140018, 1.20333, 0.127566]
        )
        < 1e-6
    )
    assert (
        np.linalg.norm(
            data_dict["field"][-1, :] - [0.0409271, 0.176052, 0.0302899]
        )
        < 1e-6
    )
    assert abs(data_dict["n_cells"] - 137980) < 1e-6
    assert abs(data_dict["field"].shape[0] - 137980) < 1e-6
    assert data_dict["name"] == "U.gas"


def test_read_unif_vec():
    """
    Test for reading uniform vectorField
    """
    case_folder = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "bird",
        "postprocess",
        "data_conditional_mean",
    )
    # Read non uniform field
    data_dict = readOFVec(
        filename=os.path.join(case_folder, "79", "U_unif_dummy")
    )
    assert np.linalg.norm(data_dict["field"] - [0.0, 0.1, 0.0]) < 1e-6
    assert data_dict["name"] == "U_unif_dummy"
