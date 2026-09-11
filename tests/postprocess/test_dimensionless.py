import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from bird.postprocess.post_quantities import (
    compute_froude_number,
    compute_instantaneous_kl,
    compute_sherwood_number,
    compute_weber_number,
    froude,
    sherwood,
    weber,
)
from bird.utilities.ofio import read_global_vars


def write_gravity_case(root, gy):
    """Write a minimal constant/g with vertical gravity gy."""
    os.makedirs(os.path.join(root, "constant"), exist_ok=True)
    with open(os.path.join(root, "constant", "g"), "w") as f:
        f.write("FoamFile\n{\n    class uniformDimensionedVectorField;\n}\n")
        f.write("dimensions      [0 1 -2 0 0 0 0];\n")
        f.write(f"value           (0 {gy} 0);\n")


def write_dimensionless_case(root, u_x, rho, sigma, cell_volumes):
    """Case with uniform U.liquid=(u_x,0,0), rho, all-liquid, and sigma."""
    os.makedirs(os.path.join(root, "0"), exist_ok=True)
    os.makedirs(os.path.join(root, "constant"), exist_ok=True)

    def write_field(name, foam_class, body):
        with open(os.path.join(root, "0", name), "w") as f:
            f.write("FoamFile\n{\n    format      ascii;\n")
            f.write(f"    class       {foam_class};\n")
            f.write(f"    object      {name};\n}}\n\n")
            f.write("dimensions      [0 0 0 0 0 0 0];\n\n")
            f.write(body)

    entries = "\n".join(f"{v:.10g}" for v in cell_volumes)
    write_field(
        "V",
        "volScalarField",
        "internalField   nonuniform List<scalar> \n"
        f"{len(cell_volumes)}\n(\n{entries}\n)\n;\n",
    )
    write_field(
        "alpha.liquid", "volScalarField", "internalField   uniform 1;\n"
    )
    write_field(
        "U.liquid",
        "volVectorField",
        f"internalField   uniform ({u_x} 0 0);\n",
    )
    write_field(
        "thermo:rho.liquid",
        "volScalarField",
        f"internalField   uniform {rho};\n",
    )
    with open(os.path.join(root, "constant", "phaseProperties"), "w") as f:
        f.write("surfaceTension\n(\n    (gas and liquid)\n    {\n")
        f.write(
            f"        type            constant;\n        sigma  {sigma};\n"
        )
        f.write("    }\n);\n")


def test_froude():
    # default gravity 9.81
    assert froude(1.0, 1.0) == pytest.approx(1.0 / np.sqrt(9.81))
    # Fr = U / sqrt(g*L); 2 / sqrt(16*0.25) = 1
    assert froude(2.0, 0.25, gravity=16.0) == pytest.approx(1.0)


def test_weber():
    # We = rho*U^2*L/sigma; 1000 * 4 * 0.5 / 100 = 20
    assert weber(1000.0, 2.0, 0.5, 100.0) == pytest.approx(20.0)


def test_sherwood():
    # Sh = kL*L/D; 1e-4 * 0.5 / 1e-5 = 5
    assert sherwood(1e-4, 0.5, 1e-5) == pytest.approx(5.0)


def test_compute_froude_number():
    # velocity = |U.liquid| = 3, g = 9 -> Fr = 3 / sqrt(9*0.25) = 2
    with tempfile.TemporaryDirectory() as tmp:
        write_dimensionless_case(
            tmp, u_x=3.0, rho=1000.0, sigma=0.05, cell_volumes=[1.0, 2.0, 3.0]
        )
        write_gravity_case(tmp, -9.0)
        fr, _ = compute_froude_number(tmp, "0", length=0.25, volume_time="0")
    assert fr == pytest.approx(2.0)


def test_compute_weber_number():
    # We = rho*U^2*L/sigma = 1000 * 9 * 0.2 / 0.05 = 36000
    with tempfile.TemporaryDirectory() as tmp:
        write_dimensionless_case(
            tmp, u_x=3.0, rho=1000.0, sigma=0.05, cell_volumes=[1.0, 2.0, 3.0]
        )
        we, _ = compute_weber_number(tmp, "0", length=0.2, volume_time="0")
    assert we == pytest.approx(1000.0 * 9.0 * 0.2 / 0.05)


def test_compute_sherwood_number():
    case_folder = os.path.join(
        Path(__file__).parent,
        " .. ".strip(),
        " .. ".strip(),
        "bird",
        "postprocess",
        "data_conditional_mean",
    )
    # Sh = kL * L / D_molecular, with kL from compute_instantaneous_kl and
    # D_CO2 read from globalVars. kL is in m/h and D in m2/s, so kL is
    # converted to m/s (/3600) for Sh to be dimensionless.
    kl_spec, _, _ = compute_instantaneous_kl(
        species_names="CO2",
        case_folder=case_folder,
        time_folder="80",
        volume_time="1",
    )
    diffusivity = float(read_global_vars(case_folder)["D_CO2"])
    sh, _ = compute_sherwood_number(
        case_folder, "80", length=0.1, species_name="CO2", volume_time="1"
    )
    assert sh == pytest.approx((kl_spec["CO2"] / 3600) * 0.1 / diffusivity)
