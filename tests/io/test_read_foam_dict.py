import os
import tempfile
from pathlib import Path

import numpy as np

from bird.utilities.ofio import read_gravity, read_openfoam_dict


def test_read_gravity():
    """
    Test reading gravity magnitude from `constant/g`, with a 9.81 fallback
    """
    case_folder = os.path.join(
        Path(__file__).parent,
        " .. ".strip(),
        " .. ".strip(),
        "tutorial_cases",
        "OF9",
        "loop_reactor_mixing_swirl",
    )
    assert read_gravity(case_folder) == 9.81

    # custom vertical gravity
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "constant"))
        with open(os.path.join(tmp, "constant", "g"), "w") as f:
            f.write("dimensions [0 1 -2 0 0 0 0];\nvalue (0 -3.0 0);\n")
        assert read_gravity(tmp) == 3.0

    # missing constant/g -> 9.81 fallback
    with tempfile.TemporaryDirectory() as tmp:
        assert read_gravity(tmp) == 9.81


def test_read_phaseProperties():
    """
    Test for reading content of `constant/phaseProperties`
    """
    const_folder = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "tutorial_cases",
        "OF9",
        "loop_reactor_mixing",
        "constant",
    )
    # Read non uniform field
    foam_dict = read_openfoam_dict(
        filename=os.path.join(const_folder, "phaseProperties")
    )

    assert foam_dict["phases"] == ["gas", "liquid"]
    assert foam_dict["gas"]["constantCoeffs"]["d"] == "3e-3"
    assert (
        foam_dict["liquid"]["Sc"]["code"]
        == "os << ( $LeLiqMix * $CpMixLiq * $muMixLiq / $kThermLiq ) ;"
    )
    assert (
        foam_dict["diffusiveMassTransfer.liquid"]["( gas in liquid )"]["type"]
        == "Higbie"
    )
    assert (
        foam_dict["lift"]["( gas in liquid )"]["lift"]["swarmCorrection"][
            "type"
        ]
        == "none"
    )


def test_read_ndf():
    """
    Test for reading content of `constant/phaseProperties` with population balance
    """
    const_folder = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "bird",
        "postprocess",
        "data_conditional_mean",
        "constant",
    )
    # Read non uniform field
    foam_dict = read_openfoam_dict(
        filename=os.path.join(const_folder, "phaseProperties")
    )

    assert foam_dict["phases"] == ["gas", "liquid"]
    assert foam_dict["populationBalances"] == ["bubbles"]
    assert foam_dict["gas"]["diameterModel"] == "velocityGroup"
    assert len(foam_dict["gas"]["velocityGroupCoeffs"]["sizeGroups"]) == 21
    assert (
        abs(
            float(
                foam_dict["gas"]["velocityGroupCoeffs"]["sizeGroups"]["f4"][
                    "dSph"
                ]
            )
            - 2.5e-3
        )
        < 1e-12
    )


def test_read_thermophysicalProperties():
    """
    Test for reading content of `constant/thermophysicalProperties`
    """
    const_folder = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "tutorial_cases",
        "OF9",
        "loop_reactor_mixing",
        "constant",
    )
    # Read non uniform field
    foam_dict = read_openfoam_dict(
        filename=os.path.join(const_folder, "thermophysicalProperties.gas")
    )

    print(foam_dict)
    assert foam_dict["species"] == ["H2", "CO2", "N2"]
    assert (
        foam_dict["CO2"]["thermodynamics"]["highCpCoeffs"][0] == "3.85746029"
    )
    assert len(foam_dict["CO2"]["thermodynamics"]["highCpCoeffs"]) == 7


def test_read_momentumTransport():
    """
    Test for reading content of `constant/momentumTransport`
    """
    const_folder = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "tutorial_cases",
        "OF9",
        "loop_reactor_mixing",
        "constant",
    )
    # Read non uniform field
    foam_dict = read_openfoam_dict(
        filename=os.path.join(const_folder, "momentumTransport.gas")
    )

    assert foam_dict["simulationType"] == "RAS"
    assert foam_dict["RAS"]["turbulence"] == "on"


def test_read_controlDict():
    """
    Test for reading content of `system/controlDict`
    """
    syst_folder = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "tutorial_cases",
        "OF9",
        "loop_reactor_mixing",
        "system",
    )
    # Read non uniform field
    foam_dict = read_openfoam_dict(
        filename=os.path.join(syst_folder, "controlDict")
    )

    assert foam_dict["writeControl"] == "adjustableRunTime"
    assert foam_dict["maxCo"] == "0.5"
