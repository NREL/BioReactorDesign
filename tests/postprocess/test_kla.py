import os

import pytest

from bird import BIRD_KLA_DATA_DIR
from bird.postprocess.kla_utils import compute_kla, print_res_dict


@pytest.mark.parametrize(
    ["bootstrap", "max_chop"],
    [
        (True, None),
        (False, 2),
    ],
)
def test_kla(bootstrap, max_chop):
    res_dict = compute_kla(
        os.path.join(BIRD_KLA_DATA_DIR, "volume_avg.dat"),
        time_ind=0,
        conc_ind=1,
        bootstrap=bootstrap,
        max_chop=max_chop,
    )
    print_res_dict(res_dict)
