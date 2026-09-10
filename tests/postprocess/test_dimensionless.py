import os
import tempfile

import numpy as np
import pytest

from bird.postprocess.post_quantities import froude, sherwood, weber


def write_gravity_case(root, gy):
    """Write a minimal constant/g with vertical gravity gy."""
    os.makedirs(os.path.join(root, "constant"), exist_ok=True)
    with open(os.path.join(root, "constant", "g"), "w") as f:
        f.write("FoamFile\n{\n    class uniformDimensionedVectorField;\n}\n")
        f.write("dimensions      [0 1 -2 0 0 0 0];\n")
        f.write(f"value           (0 {gy} 0);\n")


def test_froude():
    # no case -> default gravity 9.81
    assert froude(1.0, 1.0) == pytest.approx(1.0 / np.sqrt(9.81))

    # gravity read from constant/g: Fr = 2 / sqrt(16*0.25) = 1
    with tempfile.TemporaryDirectory() as tmp:
        write_gravity_case(tmp, -16.0)
        assert froude(2.0, 0.25, case_folder=tmp) == pytest.approx(1.0)

    # missing constant/g -> 9.81 fallback
    with tempfile.TemporaryDirectory() as tmp:
        assert froude(1.0, 1.0, case_folder=tmp) == pytest.approx(
            1.0 / np.sqrt(9.81)
        )


def test_weber():
    # We = rho*U^2*L/sigma; 1000 * 4 * 0.5 / 100 = 20
    assert weber(1000.0, 2.0, 0.5, 100.0) == pytest.approx(20.0)


def test_sherwood():
    # Sh = kL*L/D; 1e-4 * 0.5 / 1e-5 = 5
    assert sherwood(1e-4, 0.5, 1e-5) == pytest.approx(5.0)
