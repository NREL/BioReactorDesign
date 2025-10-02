import os
import shutil
from pathlib import Path

import numpy as np

from bird.utilities.ofio import get_species_name, species_name_to_mw


def test_species_mw():
    """
    Test for listing all time folders in a case
    """
    case_folder = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "bird",
        "postprocess",
        "data_conditional_mean",
    )
    species_names = get_species_name(case_folder=case_folder, phase="liquid")
    mw_species = {}
    for species_name in species_names:
        mw_species[species_name] = species_name_to_mw(
            case_folder=case_folder, species_name=species_name
        )
    assert abs(mw_species["CO2"] - 44.00995 * 1e-3) < 1e-6
    assert abs(mw_species["CO"] - 28.01055 * 1e-3) < 1e-6
    assert abs(mw_species["H2"] - 2.01594 * 1e-3) < 1e-6
    assert abs(mw_species["water"] - 18.01534 * 1e-3) < 1e-6

    shutil.move(
        os.path.join(
            case_folder, "constant", "thermophysicalProperties.liquid"
        ),
        os.path.join(case_folder, "thermophysicalProperties.liquid_tmp"),
    )
    shutil.move(
        os.path.join(case_folder, "constant", "thermophysicalProperties.gas"),
        os.path.join(case_folder, "thermophysicalProperties.gas_tmp"),
    )

    for species_name in species_names:
        mw_species[species_name] = species_name_to_mw(
            case_folder=case_folder, species_name=species_name
        )
    assert abs(mw_species["CO2"] - 0.044) < 1e-6
    assert abs(mw_species["CO"] - 0.028) < 1e-6
    assert abs(mw_species["H2"] - 0.002) < 1e-6
    assert abs(mw_species["water"] - 0.018) < 1e-6

    shutil.move(
        os.path.join(case_folder, "thermophysicalProperties.liquid_tmp"),
        os.path.join(
            case_folder, "constant", "thermophysicalProperties.liquid"
        ),
    )
    shutil.move(
        os.path.join(case_folder, "thermophysicalProperties.gas_tmp"),
        os.path.join(case_folder, "constant", "thermophysicalProperties.gas"),
    )


def test_species_names():
    """
    Make sure the species names of all the phases can be identified
    """

    case_folder = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "tutorial_cases",
        "bubble_column_20L",
    )

    gas_spec_names = get_species_name(case_folder, phase="gas")

    assert len(gas_spec_names) == 3
    assert "O2" in gas_spec_names
    assert "N2" in gas_spec_names
    assert "water" in gas_spec_names

    liq_spec_names = get_species_name(case_folder, phase="liquid")

    assert len(liq_spec_names) == 2
    assert "O2" in liq_spec_names
    assert "water" in liq_spec_names
