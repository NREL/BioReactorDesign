import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from bird.utilities.ofio import (
    _find_header_size,
    _readOF,
    _readOFScal,
    _readOFVec,
    read_bubble_diameter,
    read_mu_liquid,
    read_surface_field_value,
)


def test_read_nonunif_scal():
    """
    Test for reading non uniform scalarField
    """
    case_folder = os.path.join(
        Path(__file__).parent,
        " .. ".strip(),
        " .. ".strip(),
        "bird",
        "postprocess",
        "data_conditional_mean",
    )
    # Read non uniform field
    data_dict = _readOFScal(
        filename=os.path.join(case_folder, "79", "CO2.gas")
    )
    assert abs(data_dict["field"][0] - 0.616955) < 1e-6
    assert abs(data_dict["field"][-1] - 0.625389) < 1e-6
    assert abs(data_dict["n_cells"] - 137980) < 1e-6
    assert abs(data_dict["field"].shape[0] - 137980) < 1e-6
    assert data_dict["name"] == "CO2.gas"
    # Read non uniform field with flexible interface
    data_dict = _readOF(filename=os.path.join(case_folder, "79", "CO2.gas"))
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
        " .. ".strip(),
        " .. ".strip(),
        "bird",
        "postprocess",
        "data_conditional_mean",
    )
    # Read non uniform field
    data_dict = _readOFScal(filename=os.path.join(case_folder, "79", "f.gas"))
    assert abs(data_dict["field"] - 1) < 1e-6
    assert data_dict["n_cells"] is None
    # Read non uniform field with flexible interface
    data_dict = _readOF(filename=os.path.join(case_folder, "79", "f.gas"))
    assert abs(data_dict["field"] - 1) < 1e-6
    assert data_dict["n_cells"] is None
    # Read non uniform field with prespecified cell number
    data_dict = _readOFScal(
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
        " .. ".strip(),
        " .. ".strip(),
        "bird",
        "postprocess",
        "data_conditional_mean",
    )
    # Read non uniform field
    data_dict = _readOFVec(filename=os.path.join(case_folder, "79", "U.gas"))
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
    data_dict = _readOF(filename=os.path.join(case_folder, "79", "U.gas"))
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


def test_find_header_size():
    """
    Test for counting the header lines of a nonuniform field
    """
    case_folder = os.path.join(
        Path(__file__).parent,
        " .. ".strip(),
        " .. ".strip(),
        "bird",
        "postprocess",
        "data_conditional_mean",
    )
    n_cells = 137980
    # Scalar and vector fields of the same case share the same header layout
    scal_file = os.path.join(case_folder, "79", "CO2.gas")
    vec_file = os.path.join(case_folder, "79", "U.gas")
    assert _find_header_size(scal_file, n_cells) == 21
    assert _find_header_size(vec_file, n_cells) == 21
    # The header size is what the field readers report
    assert _readOFScal(filename=scal_file)["n_header"] == 21
    assert _readOFVec(filename=vec_file)["n_header"] == 21

    # A number of cells absent from the header never matches. Note the scan
    # looks for a substring, so a short number such as 1 would match inside
    # the real cell count
    with pytest.raises(ValueError):
        _find_header_size(vec_file, 424242)

    # Giving up early also raises rather than returning a wrong offset
    with pytest.raises(ValueError):
        _find_header_size(vec_file, n_cells, max_lines=5)


def test_read_unif_vec():
    """
    Test for reading uniform vectorField
    """
    case_folder = os.path.join(
        Path(__file__).parent,
        " .. ".strip(),
        " .. ".strip(),
        "bird",
        "postprocess",
        "data_conditional_mean",
    )
    # Read non uniform field
    data_dict = _readOFVec(
        filename=os.path.join(case_folder, "79", "U_unif_dummy")
    )
    assert np.linalg.norm(data_dict["field"] - [0.0, 0.1, 0.0]) < 1e-6
    assert data_dict["name"] == "U_unif_dummy"


def test_read_bubble_diameter():
    """
    Test for reading bubble diameter
    """
    case_folder = os.path.join(
        Path(__file__).parent,
        " .. ".strip(),
        " .. ".strip(),
        "bird",
        "postprocess",
        "data_conditional_mean",
    )
    # Read bubble diameter
    d_gas, _ = read_bubble_diameter(case_folder=case_folder, time_folder="80")
    assert abs(d_gas[2] - 0.00966694) < 1e-8
    # Read bubble diameter if no d.gas is available
    # Manufacture a case without d.gas
    shutil.move(
        os.path.join(case_folder, "80", "d.gas"),
        os.path.join(case_folder, "d.gas_tmp"),
    )
    shutil.move(
        os.path.join(case_folder, "constant", "phaseProperties"),
        os.path.join(case_folder, "phaseProperties_tmp"),
    )
    shutil.copy(
        os.path.join(case_folder, "constant", "phaseProperties_constantD"),
        os.path.join(case_folder, "constant", "phaseProperties"),
    )
    d_gas, _ = read_bubble_diameter(case_folder=case_folder, time_folder="80")
    assert abs(d_gas - 0.003) < 1e-8
    shutil.move(
        os.path.join(case_folder, "d.gas_tmp"),
        os.path.join(case_folder, "80", "d.gas"),
    )
    shutil.move(
        os.path.join(case_folder, "phaseProperties_tmp"),
        os.path.join(case_folder, "constant", "phaseProperties"),
    )
    # Read bubble diameter
    d_gas, _ = read_bubble_diameter(case_folder=case_folder, time_folder="80")
    assert abs(d_gas[2] - 0.00966694) < 1e-8


def test_read_mu_liquid():
    """
    Test for reading bubble diameter
    """
    case_folder = os.path.join(
        Path(__file__).parent,
        " .. ".strip(),
        " .. ".strip(),
        "bird",
        "postprocess",
        "data_conditional_mean",
    )
    # Read bubble diameter
    mu_liq, _ = read_mu_liquid(case_folder=case_folder, time_folder="80")
    assert abs(mu_liq - 0.001) < 1e-8
    # Read bubble diameter if no d.gas is available
    # Manufacture a case without d.gas
    shutil.move(
        os.path.join(case_folder, "80", "thermo:mu.liquid"),
        os.path.join(case_folder, "thermo:mu.liquid_tmp"),
    )
    mu_liq, _ = read_mu_liquid(case_folder=case_folder, time_folder="80")
    assert abs(mu_liq - 0.001) < 1e-8
    shutil.move(
        os.path.join(case_folder, "thermo:mu.liquid_tmp"),
        os.path.join(case_folder, "80", "thermo:mu.liquid"),
    )


def test_read_surface_field_value():
    """
    Test for reading a surfaceFieldValue.dat file
    """
    header_4 = (
        "# Region type : patch inlet\n"
        "# Faces : 100\n"
        "# Area : 0.05\n"
        "# Time            areaIntegrate(alpha.gas)\n"
    )
    data = "0               0.0123\n1               0.0456\n"

    with tempfile.TemporaryDirectory() as tmpdirname:
        # Returns the last column of the first data row
        path = os.path.join(tmpdirname, "surfaceFieldValue.dat")
        with open(path, "w") as f:
            f.write(header_4 + data)
        assert abs(read_surface_field_value(path) - 0.0123) < 1e-12

        # Robust to a different number of comment lines and to blank lines,
        # unlike a hardcoded line index
        path_6 = os.path.join(tmpdirname, "surfaceFieldValue_6.dat")
        with open(path_6, "w") as f:
            f.write("# a\n# b\n# c\n# d\n\n# e\n# f\n" + data)
        assert abs(read_surface_field_value(path_6) - 0.0123) < 1e-12

        # No data rows raises
        path_empty = os.path.join(tmpdirname, "surfaceFieldValue_empty.dat")
        with open(path_empty, "w") as f:
            f.write(header_4)
        with pytest.raises(ValueError):
            read_surface_field_value(path_empty)
