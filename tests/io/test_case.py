import os
from pathlib import Path

import numpy as np
from prettyPlot.plotting import plt, pretty_labels

from bird.utilities.ofio import getCaseTimes


def test_case_time():
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


if __name__ == "__main__":
    test_case_time()
