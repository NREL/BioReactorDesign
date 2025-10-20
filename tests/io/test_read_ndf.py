import os
from pathlib import Path

import numpy as np

from bird.utilities.ofio import read_size_groups


def test_read_size_groups():
    """
    Test for getting size group info
    """
    case_folder = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "bird",
        "postprocess",
        "data_conditional_mean",
    )
    ndf_groups = read_size_groups(case_folder)

    assert len(ndf_groups) == 21
    assert abs(ndf_groups["f4"]["diam"] - 2.5e-3) < 1e-12
    assert abs(ndf_groups["f20"]["diam"] - 10.5e-3) < 1e-12
    # Place holder for bin size
