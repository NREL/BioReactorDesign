import os

import pytest

from bird import BIRD_PRE_DATA_DIR
from bird.preprocess.json_gen.generate_designs import (
    convert_case_dim,
    replace_str_in_file,
)


@pytest.mark.parametrize(
    "dim_factor",
    [2, 4, 8],
)
def test_scale_dim(dim_factor):
    input_folder = os.path.join(BIRD_PRE_DATA_DIR, "loop_reactor_3_6L")
    convert_case_dim(input_folder, "dummy_loop", dim_factor)


if __name__ == "__main__":
    test_scale_dim(2)
