import os
import shutil
import tempfile
from pathlib import Path

from bird.preprocess.species_gen.setup_thermo_prop import (
    get_species_key_pair,
    get_species_name,
    get_species_properties,
    write_species_properties,
)
from bird.utilities.ofio import read_openfoam_dict


def test_species_thermo_write():
    """
    Artificially modify the thermo properties and make sure they are written back
    """

    case_dir = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "tutorial_cases",
        "bubble_column_20L",
    )

    # Output to temporary directory and delete when done
    with tempfile.TemporaryDirectory() as tmpdirname:
        shutil.copytree(
            os.path.join(case_dir, "constant"),
            os.path.join(tmpdirname, "constant"),
        )

        src_fake_thermo = os.path.join(
            tmpdirname, "constant", "thermophysicalProperties_erroneous.gas"
        )

        dest_thermo = os.path.join(
            tmpdirname, "constant", "thermophysicalProperties.gas"
        )

        shutil.copyfile(src_fake_thermo, dest_thermo)

        # Reading the wrong thermo file and making sure we have an error
        foam_dict = read_openfoam_dict(dest_thermo)
        assert (
            abs(
                float(foam_dict["O2"]["thermodynamics"]["highCpCoeffs"][2])
                + 7.57967e-07
            )
            > 1e-12
        )

        write_species_properties(tmpdirname, phase="gas")
        foam_dict = read_openfoam_dict(dest_thermo)
        assert (
            abs(
                float(foam_dict["O2"]["thermodynamics"]["highCpCoeffs"][2])
                + 7.57967e-07
            )
            < 1e-12
        )


def test_species_names():
    """
    Make sure the species names of all the phases can be identified
    """

    case_dir = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "tutorial_cases",
        "bubble_column_20L",
    )

    gas_spec_names = get_species_name(case_dir, phase="gas")

    assert len(gas_spec_names) == 3
    assert "O2" in gas_spec_names
    assert "N2" in gas_spec_names
    assert "water" in gas_spec_names

    liq_spec_names = get_species_name(case_dir, phase="liquid")

    assert len(liq_spec_names) == 2
    assert "O2" in liq_spec_names
    assert "water" in liq_spec_names


def test_species_prop():
    """
    Make sure the species properties are correctly read
    """

    species_list = ["O2"]
    species_prop = get_species_properties(species_list)

    assert len(species_prop) == 1
    assert "O2" in species_prop
    assert (
        abs(float(species_prop["O2"]["gas"]["transport"]["As"]) - 1.948e-06)
        < 1e-12
    )

    species_list = ["O2", "N2"]
    species_prop = get_species_properties(species_list)

    assert len(species_prop) == 2
    assert "O2" in species_prop
    assert "N2" in species_prop
    assert (
        abs(float(species_prop["O2"]["gas"]["transport"]["As"]) - 1.948e-06)
        < 1e-12
    )
    assert (
        abs(
            float(
                species_prop["N2"]["gas"]["thermodynamics"][
                    "lowCpCoeffs"
                ].split()[3]
            )
            - 5.641515e-09
        )
        < 1e-12
    )

    species_list = ["water"]
    species_prop = get_species_properties(species_list)

    assert len(species_prop) == 1
    assert "water" in species_prop
    assert (
        abs(
            float(
                species_prop["water"]["gas"]["thermodynamics"][
                    "highCpCoeffs"
                ].split()[1]
            )
            - 0.00217691804
        )
        < 1e-12
    )


def test_species_key_pair():
    """
    Make sure the species names are linked to the correct openfoam dicts
    """

    case_dir = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "experimental_cases",
        "deckwer17",
    )

    liq_spec_names = get_species_name(case_dir, phase="liquid")
    thermo_file = os.path.join(
        case_dir, "constant", "thermophysicalProperties.liquid"
    )
    foam_dict = read_openfoam_dict(thermo_file)

    species_key_pair_dict = get_species_key_pair(foam_dict, liq_spec_names)

    assert len(species_key_pair_dict) == 2
    assert "CO2" in species_key_pair_dict
    assert "water" in species_key_pair_dict
    assert species_key_pair_dict["CO2"] == "CO2"
    assert species_key_pair_dict["water"] == '"(mixture|water)"'
