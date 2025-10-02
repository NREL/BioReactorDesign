import os
from pathlib import Path

import numpy as np
import pytest

from bird.postprocess.kla_utils import compute_kla, print_res_dict


@pytest.mark.parametrize(
    ["bootstrap", "max_chop"],
    [
        (True, None),
        (False, 2),
    ],
)
def test_fitted_kla(bootstrap, max_chop):
    BIRD_KLA_DATA_FILE = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "bird",
        "postprocess",
        "data_kla",
        "volume_avg.dat",
    )
    data = np.loadtxt(BIRD_KLA_DATA_FILE)
    data_t = data[:, 0]
    data_c = data[:, 1]

    res_dict = compute_kla(
        data_t,
        data_c,
        bootstrap=bootstrap,
        max_chop=max_chop,
    )

    print_res_dict(res_dict)
