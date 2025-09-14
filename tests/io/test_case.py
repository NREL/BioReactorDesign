import os
from pathlib import Path

import numpy as np

from bird.utilities.ofio import (
    _get_mesh_time,
    _get_volume_time,
    _read_mesh,
    get_case_times,
    read_cell_centers,
    read_cell_volumes,
)


def test_case_time():
    """
    Test for listing all time folders in a case
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
    time_float, time_str = get_case_times(case_folder)
    assert np.linalg.norm(np.array(time_float) - np.array([1, 79, 80])) < 1e-6
    assert time_str == ["1", "79", "80"]


def test_mesh():
    """
    Test for reading cell centers
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
    mesh_time = _get_mesh_time(case_folder)
    mesh_file = os.path.join(case_folder, f"meshCellCentres_{mesh_time}.obj")
    cell_centers = _read_mesh(mesh_file)

    assert cell_centers[0, 0] == -0.0765
    assert cell_centers[0, 1] == 7.02306
    assert cell_centers[0, 2] == 0.0765

    assert cell_centers[9, 0] == 0.0765
    assert cell_centers[9, 1] == 7.02306
    assert cell_centers[9, 2] == 0.0765

    cell_centers2, _ = read_cell_centers(case_folder)
    cell_centers3, _ = read_cell_centers(case_folder, time_folder=mesh_time)

    assert np.linalg.norm(cell_centers - cell_centers2) < 1e-12
    assert np.linalg.norm(cell_centers - cell_centers3) < 1e-12


def test_mesh_vol():
    """
    Test for reading cell volumes
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
    vol_time = _get_volume_time(case_folder)
    volumes, _ = read_cell_volumes(case_folder)
    volumes2, _ = read_cell_volumes(case_folder, time_folder=vol_time)

    assert np.linalg.norm(volumes - volumes2) < 1e-12


if __name__ == "__main__":
    test_case_time()
    test_mesh()
