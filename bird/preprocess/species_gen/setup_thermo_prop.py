import logging
import os
import re
from collections import defaultdict

import numpy as np
from ruamel.yaml import YAML

from bird import BIRD_CONST_DIR
from bird.utilities.ofio import read_openfoam_dict, write_openfoam_dict
from bird.utilities.parser import parse_yaml

logger = logging.getLogger(__name__)


def check_phase_name(phase: str):
    """
    Check that phase name is valid

    Parameters
    ----------
    phase: str
        Name of phase where to find the species
    """
    try:
        assert phase in ["gas", "liquid"]
    except AssertionError:
        error_msg = f"Phase name ('{phase}') is not in ['gas', 'liquid']"
        logger.error(error_msg)
        raise NotImplementedError(error_msg)


def get_species_name(case_dir: str, phase: str = "gas") -> list[str]:
    """
    Get list of species name in a phase

    Parameters
    ----------
    case_dir: str
        Path to OpenFOAM case
    phase: str
        Name of phase where to find the species

    Returns
    ----------
    species_name: list[str]
        List of species name in the phase
    """
    check_phase_name(phase)
    logger.debug(f"Finding species in phase '{phase}'")

    thermo_properties = read_openfoam_dict(
        os.path.join(case_dir, "constant", f"thermophysicalProperties.{phase}")
    )

    try:
        species = thermo_properties["species"]
        if not isinstance(species, list):
            assert isinstance(species, str)
            species = [species]
    except KeyError:
        species = []
    try:
        defaultSpecie = thermo_properties["defaultSpecie"]
        if not isinstance(defaultSpecie, list):
            assert isinstance(defaultSpecie, str)
            defaultSpecie = [defaultSpecie]
    except KeyError:
        defaultSpecie = []
    try:
        inertSpecie = thermo_properties["inertSpecie"]
        if not isinstance(inertSpecie, list):
            assert isinstance(inertSpecie, str)
            inertSpecie = [inertSpecie]
    except KeyError:
        inertSpecie = []

    species_name = list(set(species + defaultSpecie + inertSpecie))
    logger.debug(f"Species in phase '{phase}' are {species_name}")
    return species_name


def get_species_properties(species_name: list[str]) -> dict:
    """
    Get thermo properties of species

    Parameters
    ----------
    species_name: list[str]
        List of species names

    Returns
    ----------
    species_prop: dict
        Dictionary of species properties
    """
    logger.debug(f"Reading properties of {species_name} from {BIRD_CONST_DIR}")
    species_prop = {}
    for species in species_name:
        if species == "water":
            species_file = "h2o.yaml"
        else:
            species_file = species.lower() + ".yaml"
        species_dict = parse_yaml(os.path.join(BIRD_CONST_DIR, species_file))
        species_prop[species] = species_dict
    return species_prop


def get_species_key_pair(
    thermo_properties: dict, species_name: list[str]
) -> dict:
    """
    Find the key where are stored the species info

    Parameters
    ----------
    thermo_properties: dict
        Dictionary of thermo properties
    species_name : list[str]
        List of species names

    Returns
    ----------
    pair_species_keys: dict
        Pair of species names and dictionary key
    """

    pair_species_keys = {}

    for species in species_name:
        if species in list(thermo_properties.keys()):
            pair_species_keys[species] = species
        else:
            found = False
            for key in list(thermo_properties.keys()):
                if species in key:
                    pair_species_keys[species] = key
                    logger.debug(f"Found '{species}' in {key}")
                    found = True
                    break
            if not found:
                thermo_properties[species] = {}
                pair_species_keys[species] = species
                logger.warning(
                    f"Could not find species '{species}' info in {thermo_properties_file}"
                )

    return pair_species_keys


def compare_old_new_prop(
    key_seq_old: list,
    key_seq_new: list,
    old_dict,
    new_dict,
    species,
    type_name="float",
):
    try:
        old_val = old_dict
        for key in key_seq_old:
            old_val = old_val[key]
        if type_name == "float":
            old_val = float(old_val)
        elif type_name == "list":
            old_val = np.array([float(entry) for entry in old_val])
        new_val = new_dict
        for key in key_seq_new:
            new_val = new_val[key]
        if type_name == "float":
            new_val = float(new_val)
        elif type_name == "list":
            new_val = new_val.replace("(", "").replace(")", "").split()
            new_val = np.array([float(entry) for entry in new_val])
        if type_name == "float":
            if not abs(old_val - new_val) < 1e-12:
                logger.warning(
                    f"{key_seq_old[-1]} of '{species}' updated from {old_val} to {new_val}"
                )
        elif type_name == "list":
            if not np.linalg.norm(old_val - new_val) < 1e-12:
                logger.warning(
                    f"{key_seq_old[-1]} of '{species}' updated from {old_val} to {new_val}"
                )
        return "success"
    except KeyError:
        return "failure"


def update_gas_thermo_prop(
    thermo_properties: dict, species_prop: dict, pair_species_keys: dict
) -> dict:
    """
    Update gas thermo properties

    Parameters
    ----------
    thermo_properties: dict
        Dictionary of thermo properties
    species_prop : dict
        Dictionary of species properties
    pair_species_keys: dict
        Dictionary that maps species name to thermo properties key

    Returns
    ----------
    thermo_properties : dict
        Updated gas thermo properties
    """
    species_name = list(pair_species_keys.keys())

    for species in species_name:
        key_val = pair_species_keys[species]
        spec_dict = thermo_properties[key_val]
        # Molecular weight
        target_wm_dict = {
            "molWeight": str(species_prop[species]["specie"]["molWeight"])
        }
        if not "specie" in spec_dict:
            spec_dict["specie"] = target_wm_dict
        else:
            status = compare_old_new_prop(
                key_seq_old=["specie", "molWeight"],
                key_seq_new=["molWeight"],
                old_dict=spec_dict,
                new_dict=target_wm_dict,
                species=species,
                type_name="float",
            )
            if status == "failure":
                spec_dict["specie"] = target_wm_dict

        # Thermo coeff
        target_thermo_dict = {
            "Tlow": str(
                species_prop[species]["gas"]["thermodynamics"]["Tlow"]
            ),
            "Thigh": str(
                species_prop[species]["gas"]["thermodynamics"]["Thigh"]
            ),
            "highCpCoeffs": "( "
            + str(
                species_prop[species]["gas"]["thermodynamics"]["highCpCoeffs"]
            )
            + " )",
            "lowCpCoeffs": "( "
            + str(
                species_prop[species]["gas"]["thermodynamics"]["lowCpCoeffs"]
            )
            + " )",
        }
        if "Tcommon" in species_prop[species]["gas"]["thermodynamics"]:
            target_thermo_dict["Tcommon"] = str(
                species_prop[species]["gas"]["thermodynamics"]["Tcommon"]
            )
            thermo_keys = ["Tlow", "Thigh", "Tcommon"]
        else:
            thermo_keys = ["Tlow", "Thigh"]

        if not "thermodynamics" in spec_dict:
            spec_dict["thermodynamics"] = target_thermo_dict
        else:
            for thermo_key in thermo_keys:
                status = compare_old_new_prop(
                    key_seq_old=["thermodynamics", thermo_key],
                    key_seq_new=[thermo_key],
                    old_dict=spec_dict,
                    new_dict=target_thermo_dict,
                    species=species,
                    type_name="float",
                )
                if status == "failure":
                    spec_dict["thermodynamic"][thermo_key] = (
                        target_thermo_dict[thermo_key]
                    )
            for thermo_key in ["highCpCoeffs", "lowCpCoeffs"]:
                status = compare_old_new_prop(
                    key_seq_old=["thermodynamics", thermo_key],
                    key_seq_new=[thermo_key],
                    old_dict=spec_dict,
                    new_dict=target_thermo_dict,
                    species=species,
                    type_name="list",
                )
                if status == "failure":
                    spec_dict["thermodynamic"][thermo_key] = (
                        target_thermo_dict[thermo_key]
                    )

        # Transport coeff
        target_transport_dict = {
            "As": str(species_prop[species]["gas"]["transport"]["As"]),
            "Ts": str(species_prop[species]["gas"]["transport"]["Ts"]),
        }
        if not "transport" in spec_dict:
            spec_dict["transport"] = target_transport_dict
        else:
            for trans_key in ["As", "Ts"]:
                status = compare_old_new_prop(
                    key_seq_old=["transport", trans_key],
                    key_seq_new=[trans_key],
                    old_dict=spec_dict,
                    new_dict=target_transport_dict,
                    species=species,
                    type_name="float",
                )
                if status == "failure":
                    spec_dict["transport"][trans_key] = target_transport_dict[
                        trans_key
                    ]

        # Elements
        target_element_dict = species_prop[species]["specie"]["elements"]
        if not "elements" in spec_dict:
            spec_dict["elements"] = target_element_dict

        thermo_properties[key_val] = spec_dict

    return thermo_properties


def update_liq_thermo_prop(
    thermo_properties: dict, species_prop: dict, pair_species_keys: dict
) -> dict:
    """
    Update liquid thermo properties

    Parameters
    ----------
    thermo_properties: dict
        Dictionary of thermo properties
    species_prop : dict
        Dictionary of species properties
    pair_species_keys: dict
        Dictionary that maps species name to thermo properties key

    Returns
    ----------
    thermo_properties : dict
        Updated liquid thermo properties
    """

    species_name = list(pair_species_keys.keys())
    for species in species_name:
        key_val = pair_species_keys[species]
        spec_dict = thermo_properties[key_val]

        # Molecular weight
        target_wm_dict = {
            "molWeight": str(species_prop[species]["specie"]["molWeight"])
        }
        if not "specie" in spec_dict:
            spec_dict["specie"] = target_wm_dict
        else:
            status = compare_old_new_prop(
                key_seq_old=["specie", "molWeight"],
                key_seq_new=["molWeight"],
                old_dict=spec_dict,
                new_dict=target_wm_dict,
                species=species,
                type_name="float",
            )
            if status == "failure":
                spec_dict["specie"] = target_wm_dict

        # Thermo coeff
        target_thermo_dict = {
            "Cv": "$CpMixLiq",
            "Hf": species_prop[species]["liquid"]["thermodynamics"]["Hf"],
        }
        if not "thermodynamics" in spec_dict:
            spec_dict["thermodynamics"] = target_thermo_dict
        else:
            for thermo_key in ["Hf"]:
                status = compare_old_new_prop(
                    key_seq_old=["thermodynamics", thermo_key],
                    key_seq_new=[thermo_key],
                    old_dict=spec_dict,
                    new_dict=target_thermo_dict,
                    species=species,
                    type_name="float",
                )

        thermo_properties[key_val] = spec_dict

    return thermo_properties


def write_species_properties(case_dir: str, phase: str = "gas") -> None:
    """
    Write thermo properties open foam dict

    Parameters
    ----------
    case_dir: str
        Path to OpenFOAM case
    phase: str
        Name of phase where to find the species
    """
    logger.debug(f"Writing properties for phase '{phase}' in case {case_dir}")
    species_name = get_species_name(case_dir=case_dir, phase=phase)
    species_prop = get_species_properties(species_name)
    thermo_properties_file = os.path.join(
        case_dir, "constant", f"thermophysicalProperties.{phase}"
    )
    thermo_properties = read_openfoam_dict(thermo_properties_file)
    pair_species_keys = get_species_key_pair(
        thermo_properties=thermo_properties, species_name=species_name
    )
    if phase == "gas":
        thermo_properties_update = update_gas_thermo_prop(
            thermo_properties, species_prop, pair_species_keys
        )
    if phase == "liquid":
        thermo_properties_update = update_liq_thermo_prop(
            thermo_properties, species_prop, pair_species_keys
        )
    filename = os.path.join(
        case_dir, "constant", f"thermophysicalProperties.{phase}"
    )
    write_openfoam_dict(thermo_properties_update, filename=filename)


# def write_thermo_properties(case_dir:str, phase:str="gas") -> None:


if __name__ == "__main__":
    from bird import BIRD_DIR

    case_dir = os.path.join(BIRD_DIR, "../experimental_cases/deckwer17")
    write_species_properties(case_dir, phase="gas")
    write_species_properties(case_dir, phase="liquid")
    # fill_global_prop(os.path.join(BIRD_DIR,"../experimental_cases_new/disengagement/bubble_column_pbe_20L/"))
    # fill_global_prop(os.path.join(BIRD_DIR, "../experimental_cases_new/deckwer17"))
