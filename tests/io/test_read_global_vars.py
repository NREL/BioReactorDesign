import os
from pathlib import Path

import numpy as np

from bird.utilities.ofio import read_global_vars


def test_read_global_vars():
    """
    Test for reading content of `constant/globalVars`
    """
    case_folder = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "tutorial_cases",
        "loop_reactor_mixing",
    )
    # Read globalVars from case_folder path
    globalVars_dict = read_global_vars(
        case_folder=case_folder, cross_ref=False
    )
    assert globalVars_dict["T0"] == 300
    assert (
        globalVars_dict["muMixLiq"]
        == '#calc "2.414e-5 * pow(10,247.8/($T0 - 140.0))"'
    )
    assert (
        globalVars_dict["D_H2"]
        == '#calc "1.173e-16 * pow($WC_psi * $WC_M,0.5) * $T0 / $muMixLiq / pow($WC_V_H2,0.6)"'
    )
    assert (
        globalVars_dict["kN2"]
        == '#calc "$D_N2*$rho0MixLiq*$CpMixLiq*$LeLiqMix"'
    )
    assert (
        globalVars_dict["He_CO2"]
        == '#calc "$H_CO2_298 * exp($DH_CO2 *(1. / $T0 - 1./298.15))"'
    )
    assert globalVars_dict["intensity"] == 0.05

    # Read globalVars from globalVars filename
    globalVars_dict = read_global_vars(
        filename=os.path.join(case_folder, "constant", "globalVars"),
        cross_ref=False,
    )
    assert globalVars_dict["T0"] == 300
    assert (
        globalVars_dict["muMixLiq"]
        == '#calc "2.414e-5 * pow(10,247.8/($T0 - 140.0))"'
    )
    assert (
        globalVars_dict["D_H2"]
        == '#calc "1.173e-16 * pow($WC_psi * $WC_M,0.5) * $T0 / $muMixLiq / pow($WC_V_H2,0.6)"'
    )
    assert (
        globalVars_dict["kN2"]
        == '#calc "$D_N2*$rho0MixLiq*$CpMixLiq*$LeLiqMix"'
    )
    assert (
        globalVars_dict["He_CO2"]
        == '#calc "$H_CO2_298 * exp($DH_CO2 *(1. / $T0 - 1./298.15))"'
    )
    assert globalVars_dict["intensity"] == 0.05

    # Make sure correct error is raised if necessary
    try:
        globalVars_dict = read_global_vars(case_folder=None, filename=None)
    except FileNotFoundError:
        pass
    try:
        globalVars_dict = read_global_vars()
    except FileNotFoundError:
        pass
    try:
        globalVars_dict = read_global_vars(
            case_folder=None, filename="garbage"
        )
    except FileNotFoundError:
        pass
    try:
        globalVars_dict = read_global_vars(case_folder="garbage")
    except FileNotFoundError:
        pass

    # Test cross referencing
    globalVars_dict = read_global_vars(case_folder=case_folder)
    assert (
        abs(abs(globalVars_dict["muMixLiq"] - 0.0008540578) / 0.0008540578)
        < 1e-3
    )
    assert (
        abs(
            abs(globalVars_dict["D_CO2"] - 2.1437176189840984e-09)
            / 2.1437176189840984e-09
        )
        < 1e-3
    )
    assert (
        abs(
            abs(globalVars_dict["He_CO2"] - 0.789805140591798)
            / 0.789805140591798
        )
        < 1e-3
    )
