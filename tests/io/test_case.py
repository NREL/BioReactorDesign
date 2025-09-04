import os
from pathlib import Path

import numpy as np

from bird.utilities.ofio import getCaseTimes, getMeshTime, readMesh


def test_case_time():
    """
    Test for listing all time folders in a case
    """
    caseFolder = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "bird",
        "postprocess",
        "data_conditional_mean",
    )
    # Read non uniform field
    time_float, time_str = getCaseTimes(caseFolder)
    assert np.linalg.norm(np.array(time_float) - np.array([1, 79, 80])) < 1e-6
    assert time_str == ["1", "79", "80"]


def test_mesh():
    """
    Test for reading cell centers
    """
    caseFolder = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "bird",
        "postprocess",
        "data_conditional_mean",
    )
    # Read non uniform field
    mesh_time = getMeshTime(caseFolder)
    mesh_file = os.path.join(caseFolder, f"meshCellCentres_{mesh_time}.obj")
    cell_centers = readMesh(mesh_file)

    assert cell_centers[0, 0] == -0.0765
    assert cell_centers[0, 1] == 7.02306
    assert cell_centers[0, 2] == 0.0765

    assert cell_centers[9, 0] == 0.0765
    assert cell_centers[9, 1] == 7.02306
    assert cell_centers[9, 2] == 0.0765


if __name__ == "__main__":
    test_case_time()
    test_mesh()
