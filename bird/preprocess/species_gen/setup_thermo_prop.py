import os
import re
from collections import defaultdict

import numpy as np
from ruamel.yaml import YAML

from bird import BIRD_CONST_DIR, logger
from bird.utilities.ofio import (
    get_species_name,
    read_openfoam_dict,
    write_openfoam_dict,
)
from bird.utilities.parser import parse_yaml


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
    for spec_name in species_name:
        if spec_name == "water":
            species_file = "h2o.yaml"
        else:
            species_file = spec_name.lower() + ".yaml"
        species_dict = parse_yaml(os.path.join(BIRD_CONST_DIR, species_file))
        species_prop[spec_name] = species_dict
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

    for spec_name in species_name:
        if spec_name in list(thermo_properties.keys()):
            pair_species_keys[spec_name] = spec_name
        else:
            found = False
            for key in list(thermo_properties.keys()):
                if spec_name in key:
                    pair_species_keys[spec_name] = key
                    logger.debug(f"Found '{spec_name}' in {key}")
                    found = True
                    break
            if not found:
                thermo_properties[spec_name] = {}
                pair_species_keys[spec_name] = spec_name
                logger.warning(
                    f"Could not find species '{spec_name}' info in {thermo_properties_file}"
                )

    return pair_species_keys


def compare_old_new_prop(
    key_seq_old: list[str],
    key_seq_new: list[str],
    old_dict: dict,
    new_dict: dict,
    spec_name: str,
    phase: str,
    type_name="float",
) -> str:
    """
    Compare old and new thermo properties and report any mismatch

    Parameters
    ----------
    key_seq_old: list[str]
        Sequence of keys in the old dict that gives access to the old property value
    key_seq_new: list[str]
        Sequence of keys in the new dict that gives access to the new property value
    old_dict: dict
        Old dictionary of values
    new_dict: dict
        New dictionary of values
    spec_name: str
        Name of the species (useful for reporting mismatch)
    phase: str
        Name of the phase (useful for reporting mismatch)
    type_name: str
        Name of the data type compared (useful for detecting mismatch).
        Can be 'float' or 'list' for now

    Returns
    ----------
    status: str
        Whether comparison was successful or not, allows for handling missing keys
    """

    assert type_name in ["float", "list"]

    status = ""
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
                    f"{key_seq_old[-1]} of '{spec_name}' ({phase}) updated from {old_val} to {new_val}"
                )
        elif type_name == "list":
            if not np.linalg.norm(old_val - new_val) < 1e-12:
                logger.warning(
                    f"{key_seq_old[-1]} of '{spec_name}' ({phase}) updated from {old_val} to {new_val}"
                )
        status = "success"
    except KeyError:
        status = "failure"
    return status


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

    for spec_name in species_name:
        key_val = pair_species_keys[spec_name]
        spec_dict = thermo_properties[key_val]
        # Molecular weight
        target_wm_dict = {
            "molWeight": str(species_prop[spec_name]["specie"]["molWeight"])
        }
        if not "specie" in spec_dict:
            spec_dict["specie"] = target_wm_dict
        else:
            status = compare_old_new_prop(
                key_seq_old=["specie", "molWeight"],
                key_seq_new=["molWeight"],
                old_dict=spec_dict,
                new_dict=target_wm_dict,
                spec_name=spec_name,
                phase="gas",
                type_name="float",
            )
            spec_dict["specie"]["molWeight"] = target_wm_dict["molWeight"]
            if status.lower() == "failure":
                spec_dict["specie"] = target_wm_dict

        # Thermo coeff
        thermo_keys = []
        thermo_keys_cp_coeff = []
        target_thermo_dict = {}
        if "Tlow" in species_prop[spec_name]["gas"]["thermodynamics"]:
            target_thermo_dict["Tlow"] = str(
                species_prop[spec_name]["gas"]["thermodynamics"]["Tlow"]
            )
            thermo_keys.append("Tlow")
        if "Thigh" in species_prop[spec_name]["gas"]["thermodynamics"]:
            target_thermo_dict["Thigh"] = str(
                species_prop[spec_name]["gas"]["thermodynamics"]["Thigh"]
            )
            thermo_keys.append("Thigh")
        if "Tcommon" in species_prop[spec_name]["gas"]["thermodynamics"]:
            target_thermo_dict["Tcommon"] = str(
                species_prop[spec_name]["gas"]["thermodynamics"]["Tcommon"]
            )
            thermo_keys.append("Tcommon")
        if "highCpCoeffs" in species_prop[spec_name]["gas"]["thermodynamics"]:
            target_thermo_dict["highCpCoeffs"] = (
                "( "
                + str(
                    species_prop[spec_name]["gas"]["thermodynamics"][
                        "highCpCoeffs"
                    ]
                )
                + " )"
            )
            thermo_keys_cp_coeff.append("highCpCoeffs")
        if "lowCpCoeffs" in species_prop[spec_name]["gas"]["thermodynamics"]:
            target_thermo_dict["lowCpCoeffs"] = (
                "( "
                + str(
                    species_prop[spec_name]["gas"]["thermodynamics"][
                        "lowCpCoeffs"
                    ]
                )
                + " )"
            )
            thermo_keys_cp_coeff.append("lowCpCoeffs")

        if not "thermodynamics" in spec_dict:
            spec_dict["thermodynamics"] = target_thermo_dict
        else:
            for thermo_key in thermo_keys:
                status = compare_old_new_prop(
                    key_seq_old=["thermodynamics", thermo_key],
                    key_seq_new=[thermo_key],
                    old_dict=spec_dict,
                    new_dict=target_thermo_dict,
                    spec_name=spec_name,
                    phase="gas",
                    type_name="float",
                )
                spec_dict["thermodynamics"][thermo_key] = target_thermo_dict[
                    thermo_key
                ]

            for thermo_key in thermo_keys_cp_coeff:
                status = compare_old_new_prop(
                    key_seq_old=["thermodynamics", thermo_key],
                    key_seq_new=[thermo_key],
                    old_dict=spec_dict,
                    new_dict=target_thermo_dict,
                    spec_name=spec_name,
                    phase="gas",
                    type_name="list",
                )
                spec_dict["thermodynamics"][thermo_key] = target_thermo_dict[
                    thermo_key
                ]

        # Transport coeff
        transport_keys = []
        target_transport_dict = {}
        if "As" in species_prop[spec_name]["gas"]["transport"]:
            target_transport_dict["As"] = str(
                species_prop[spec_name]["gas"]["transport"]["As"]
            )
            transport_keys.append("As")
        if "Ts" in species_prop[spec_name]["gas"]["transport"]:
            target_transport_dict["Ts"] = str(
                species_prop[spec_name]["gas"]["transport"]["Ts"]
            )
            transport_keys.append("Ts")
        if not "transport" in spec_dict:
            spec_dict["transport"] = target_transport_dict
        else:
            for trans_key in transport_keys:
                status = compare_old_new_prop(
                    key_seq_old=["transport", trans_key],
                    key_seq_new=[trans_key],
                    old_dict=spec_dict,
                    new_dict=target_transport_dict,
                    spec_name=spec_name,
                    phase="gas",
                    type_name="float",
                )
                spec_dict["transport"][trans_key] = target_transport_dict[
                    trans_key
                ]

        # Elements
        target_element_dict = species_prop[spec_name]["specie"]["elements"]
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
    for spec_name in species_name:
        key_val = pair_species_keys[spec_name]
        spec_dict = thermo_properties[key_val]

        # Molecular weight
        target_wm_dict = {
            "molWeight": str(species_prop[spec_name]["specie"]["molWeight"])
        }
        if not "specie" in spec_dict:
            spec_dict["specie"] = target_wm_dict
        else:
            status = compare_old_new_prop(
                key_seq_old=["specie", "molWeight"],
                key_seq_new=["molWeight"],
                old_dict=spec_dict,
                new_dict=target_wm_dict,
                spec_name=spec_name,
                phase="liquid",
                type_name="float",
            )
            spec_dict["specie"]["molWeight"] = target_wm_dict["molWeight"]
            if status.lower() == "failure":
                spec_dict["specie"] = target_wm_dict

        # Thermo coeff
        target_thermo_dict = {
            "Cv": "$CpMixLiq",
            "Hf": species_prop[spec_name]["liquid"]["thermodynamics"]["Hf"],
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
                    spec_name=spec_name,
                    phase="liquid",
                    type_name="float",
                )
                spec_dict["thermodynamics"][thermo_key] = target_thermo_dict[
                    thermo_key
                ]

        thermo_properties[key_val] = spec_dict

    return thermo_properties


def write_species_properties(case_folder: str, phase: str = "gas") -> None:
    """
    Write thermo properties open foam dict

    Parameters
    ----------
    case_folder: str
        Path to OpenFOAM case
    phase: str
        Name of phase where to find the species
    """
    logger.debug(
        f"Writing properties for phase '{phase}' in case {case_folder}"
    )
    species_name = get_species_name(case_folder=case_folder, phase=phase)
    species_prop = get_species_properties(species_name)
    thermo_properties_file = os.path.join(
        case_folder, "constant", f"thermophysicalProperties.{phase}"
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
        case_folder, "constant", f"thermophysicalProperties.{phase}"
    )
    write_openfoam_dict(thermo_properties_update, filename=filename)
