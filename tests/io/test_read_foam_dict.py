import os
from pathlib import Path

import numpy as np

from bird.utilities.ofio import parse_openfoam_dict


def test_read_phaseProperties():
    const_folder = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "tutorial_cases",
        "loop_reactor_mixing",
        "constant",
    )
    # Read non uniform field
    foam_dict = parse_openfoam_dict(
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


def test_read_thermophysicalProperties():
    const_folder = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "tutorial_cases",
        "loop_reactor_mixing",
        "constant",
    )
    # Read non uniform field
    foam_dict = parse_openfoam_dict(
        filename=os.path.join(const_folder, "thermophysicalProperties.gas")
    )

    assert foam_dict["species"] == ["H2", "CO2", "N2"]
    assert (
        foam_dict["CO2"]["thermodynamics"]["highCpCoeffs"][0] == "3.85746029"
    )
    assert len(foam_dict["CO2"]["thermodynamics"]["highCpCoeffs"]) == 7


def test_read_momentumTransport():
    const_folder = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "tutorial_cases",
        "loop_reactor_mixing",
        "constant",
    )
    # Read non uniform field
    foam_dict = parse_openfoam_dict(
        filename=os.path.join(const_folder, "momentumTransport.gas")
    )

    assert foam_dict["simulationType"] == "RAS"
    assert foam_dict["RAS"]["turbulence"] == "on"

def test_read_controlDict():
    syst_folder = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "tutorial_cases",
        "loop_reactor_mixing",
        "system",
    )
    # Read non uniform field
    foam_dict = parse_openfoam_dict(
        filename=os.path.join(syst_folder, "controlDict")
    )

    assert foam_dict["writeControl"] == "adjustableRunTime"
    assert foam_dict["maxCo"] == "0.5"

if __name__ == "__main__":
    test_read_phaseProperties()
    test_read_thermophysicalProperties()
    test_read_momentumTransport()
    test_read_controlDict()
