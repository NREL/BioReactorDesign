import os

import numpy as np

from bird import logger

from .foam_dict_io import read_openfoam_dict
from .foam_field_io import read_field
from .global_vars import read_global_vars


def _check_phase_name(phase: str):
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


def get_species_name(case_folder: str, phase: str = "gas") -> list[str]:
    """
    Get list of species name in a phase

    Parameters
    ----------
    case_folder: str
        Path to OpenFOAM case
    phase: str
        Name of phase where to find the species

    Returns
    ----------
    species_name: list[str]
        List of species name in the phase
    """
    _check_phase_name(phase)
    logger.debug(f"Finding species in phase '{phase}'")

    thermo_properties = read_openfoam_dict(
        os.path.join(
            case_folder, "constant", f"thermophysicalProperties.{phase}"
        )
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


def read_mu_liquid(
    case_folder: str,
    time_folder: str | None = None,
    n_cells: int | None = None,
    field_dict: dict | None = None,
) -> tuple[np.ndarray | float, dict]:
    """
    Read liquid viscosity at a given time and store it in dictionary for later reuse.
    A specific function is constructed so that if thermo:mu.liquid is not available, the liquid viscosity is read from globalVars

    Parameters
    ----------
    case_folder: str
        Path to case folder
    time_folder: str | None
        Name of time folder to analyze.
        If None, it will be found automatically
    n_cells : int | None
        Number of cells in the domain.
        If None, it will deduced from the field reading
    field_dict : dict | None
        Dictionary of fields used to avoid rereading the same fields to calculate different quantities

    Returns
    ----------
    mu_liq : np.ndarray | float
        Field of liquid viscosity
    field_dict : dict
        Dictionary of fields read
    """

    if field_dict is None:
        field_dict = {}

    kwargs = {
        "case_folder": case_folder,
        "time_folder": time_folder,
        "n_cells": n_cells,
    }

    if (
        not ("thermo:mu.liquid" in field_dict)
        or field_dict["thermo:mu.liquid"] is None
    ):
        try:
            mu_liq, field_dict = read_field(
                field_name="thermo:mu.liquid", field_dict=field_dict, **kwargs
            )

        except FileNotFoundError as err:
            logger.debug(
                "Could not find thermo:mu.liquid, checking if it can be read from globalVars"
            )
            # thermo:mu.liquid does not exist
            mu_liq = None
            globalVars = read_global_vars(case_folder, cross_ref=True)
            thermo = read_openfoam_dict(
                os.path.join(
                    case_folder, "constant", "thermophysicalProperties.liquid"
                )
            )
            liquid_species = get_species_name(case_folder, phase="liquid")
            main_liq_species = None
            for species_name in liquid_species:
                if "water" in liquid_species:
                    main_liq_species = "water"
                    break
                if "WATER" in liquid_species:
                    main_liq_species = "WATER"
                    break
                if "h2o" in liquid_species:
                    main_liq_species = "h2o"
                    break
                if "H2O" in liquid_species:
                    main_liq_species = "H2O"
                    break
            for key in thermo:
                if main_liq_species in key:
                    liq_species_prop = thermo[key]
                    break
            if liq_species_prop["transport"]["mu"] == "$muMixLiq":
                # You are using a constant mu
                mu_liq = globalVars["muMixLiq"]
            if mu_liq is None:
                logger.error(
                    "Liquid viscosity is not constant, yet could not find thermo:mu.liquid"
                )
                raise FileNotFoundError(err)
            else:
                # Get the constant bubble diameter
                logger.debug("Reading liquid viscosity from globalVars")
                field_dict["thermo:mu.liquid"] = mu_liq

    else:
        # Get field from dict if it has been read before
        mu_liq = field_dict["thermo:mu.liquid"]

    return mu_liq, field_dict


def _mw_from_specie_dict(spec_dict: dict) -> float | None:
    """
    Molecular weight in kg/mol from a specie entry, None if unreadable

    This is the only place where the molar weight of OpenFOAM dictionaries
    (in g/mol) is converted to kg/mol.

    Parameters
    ----------
    spec_dict: dict
        Entry of a thermophysicalProperties dictionary for a single species

    Returns
    -------
    mw: float | None
        Molecular weight in kg/mol, or None if the entry has no molWeight
    """
    try:
        mw_g_per_mol = float(spec_dict["specie"]["molWeight"])
    except (KeyError, TypeError, ValueError):
        return None

    # HACK to make sure the right units are passed. OpenFOAM writes molWeight
    # in g/mol, where every species is heavier than 1 (the lightest one, H2,
    # is 2.016 g/mol). A value below 1 means the case already holds kg/mol,
    # and the conversion below would then be applied a second time. Fail
    # loudly rather than return a molecular weight wrong by a factor of 1000
    assert (
        mw_g_per_mol > 1.0
    ), f"molWeight {mw_g_per_mol} is below 1, so it looks like kg/mol while g/mol is expected"

    return mw_g_per_mol * 1e-3


def _mw_from_thermo_file(
    filename: str, species_names: list[str]
) -> float | None:
    """
    Molecular weight in kg/mol read from a thermophysicalProperties file

    Parameters
    ----------
    filename: str
        Path to a thermophysicalProperties file
    species_names: list[str]
        Candidate spellings of the species name, tried in order

    Returns
    -------
    mw: float | None
        Molecular weight in kg/mol, or None if no candidate name was found
    """
    thermo = read_openfoam_dict(filename)

    for name in species_names:
        if name in thermo:
            mw = _mw_from_specie_dict(thermo[name])
            if mw is not None:
                return mw

    # Several species may share an entry, with a key such as "(mixture|water)"
    for name in species_names:
        for key in thermo:
            if name in key:
                mw = _mw_from_specie_dict(thermo[key])
                if mw is not None:
                    return mw

    return None


def species_name_to_mw(case_folder: str, species_name: str) -> float:
    r"""
    Get molecular weight in :math:`kg/mol` from the species name.
    In order of availability, the molecular weight is read from the thermophysicalProperties.XXX,
    and globalVars.
    If nothing is found in globalVars (last resort) an error is raised.
    This function is primarily useful to compute concentrations

    Parameters
    ----------
    case_folder: str
        Path to case folder
    species_name: str
        The name of the species for which molecular weight is desired

    Returns
    ----------
    mw : float
        The species molecular weight
    """
    thermo_gas_filename = os.path.join(
        case_folder, "constant", "thermophysicalProperties.gas"
    )
    thermo_liq_filename = os.path.join(
        case_folder, "constant", "thermophysicalProperties.liquid"
    )
    globalVars_filename = os.path.join(case_folder, "constant", "globalVars")

    # Handle corner case of len 1 list of strings, and interpret it as string
    if (
        isinstance(species_name, list)
        and len(species_name) == 1
        and isinstance(species_name[0], str)
    ):
        species_name = species_name[0]
    assert isinstance(species_name, str)

    # Special case for water
    if species_name.lower() in ["water", "h2o"]:
        species_names = ["water", "h2o", "H2O", "WATER"]
    else:
        species_names = [
            species_name,
            species_name.upper(),
            species_name.lower(),
        ]

    mw = None
    # Try finding the molecular weight from the thermo files, liquid first
    for phase, thermo_filename in (
        ("liquid", thermo_liq_filename),
        ("gas", thermo_gas_filename),
    ):
        if mw is not None or not os.path.isfile(thermo_filename):
            continue
        mw = _mw_from_thermo_file(thermo_filename, species_names)
        if mw is not None:
            logger.debug(
                f"Read the {species_name} molecular weight ({mw}) from thermophysicalProperties.{phase}"
            )
        else:
            logger.debug(
                f"Could not read the {species_name} molecular weight from thermophysicalProperties.{phase}"
            )

    # Last resort: try finding the molecular weight from globalVars
    if mw is None and os.path.isfile(globalVars_filename):
        globalVars = read_global_vars(case_folder=case_folder, cross_ref=True)
        for name in species_names:
            if f"Mw_{name}" in globalVars:
                mw = float(globalVars[f"Mw_{name}"])
                break

    if mw is None:
        err_msg = f"Could not find the molecular weight of {species_name}"
        err_msg += f" in case {case_folder}."
        err_msg += f"\nIf you add Mw_{species_name} to globalVars,"
        err_msg += " it should be [kg/mol]"
        raise KeyError(err_msg)

    return mw
