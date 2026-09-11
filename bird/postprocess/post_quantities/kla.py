import numpy as np

from bird import logger
from bird.utilities.ofio import (
    get_case_times,
    read_bubble_diameter,
    read_cell_centers,
    read_cell_volumes,
    read_field,
    read_global_vars,
    read_mu_liquid,
    species_name_to_mw,
)

from ..kla_utils import compute_kla
from ._cell_filter import _field_filter, _get_ind_liq, _weighted_average
from .phase import (
    _read_liquid_density_field,
    compute_ave_bubble_diam,
    compute_gas_holdup,
    interfacial_area,
)
from .species import compute_ave_conc_liq


def _instantaneous_kl_field(
    case_folder: str,
    time_folder: str,
    species_names: str | list[str],
    n_cells: int | None = None,
    field_dict: dict | None = None,
) -> tuple[dict, dict]:
    """Per-cell mass-transfer coefficient kL over the liquid, per species."""
    if field_dict is None:
        field_dict = {}
    if isinstance(species_names, str):
        species_names = [species_names]

    kwargs = {
        "case_folder": case_folder,
        "time_folder": time_folder,
        "n_cells": n_cells,
    }
    globalVars = read_global_vars(case_folder=case_folder, cross_ref=True)
    ind_liq, field_dict = _get_ind_liq(field_dict=field_dict, **kwargs)

    rho_liq, field_dict = _read_liquid_density_field(
        case_folder, time_folder, n_cells, field_dict
    )
    U_gas, field_dict = read_field(
        field_name="U.gas", field_dict=field_dict, **kwargs
    )
    U_liq, field_dict = read_field(
        field_name="U.liquid", field_dict=field_dict, **kwargs
    )
    d_gas, field_dict = read_bubble_diameter(field_dict=field_dict, **kwargs)
    mu_liq, field_dict = read_mu_liquid(field_dict=field_dict, **kwargs)

    rho_liq = _field_filter(rho_liq, ind=ind_liq, field_type="scalar")
    U_gas = _field_filter(U_gas, ind=ind_liq, field_type="vector")
    U_liq = _field_filter(U_liq, ind=ind_liq, field_type="vector")
    d_gas = _field_filter(d_gas, ind=ind_liq, field_type="scalar")
    mu_liq = _field_filter(mu_liq, ind=ind_liq, field_type="scalar")

    # Magnitude of the slip velocity. Using the last axis keeps this valid
    # whether the velocities are uniform, shape (3,), or per cell, shape (N,3)
    mag_U_diff = np.linalg.norm(U_gas - U_liq, axis=-1)
    Re = rho_liq * mag_U_diff * d_gas / mu_liq

    kl_spec_field = {}
    for species_name in species_names:
        if not f"D_{species_name}" in globalVars:
            err_msg = f"D_{species_name} was not found in globalVars."
            err_msg += f'\nIf you add it, it should be looking like #calc "1.173e-16 * pow($WC_psi * $WC_M,0.5) * $T0 / $muMixLiq / pow($WC_V_{species_name},0.6)";'
            raise KeyError(err_msg)
        kl_spec_field[species_name] = (
            (2 / np.pi**0.5)
            * 3600
            * (Re**0.5)
            * (((mu_liq / rho_liq) / globalVars[f"D_{species_name}"]) ** 0.5)
            * (globalVars[f"D_{species_name}"] / d_gas)
        )
    return kl_spec_field, field_dict


def _instantaneous_a_field(
    case_folder: str,
    time_folder: str,
    n_cells: int | None = None,
    field_dict: dict | None = None,
) -> tuple[np.ndarray | float, dict]:
    """Per-cell interfacial area a = 6 alpha_gas / d over the liquid."""
    if field_dict is None:
        field_dict = {}
    kwargs = {
        "case_folder": case_folder,
        "time_folder": time_folder,
        "n_cells": n_cells,
    }
    ind_liq, field_dict = _get_ind_liq(field_dict=field_dict, **kwargs)
    alpha_gas, field_dict = read_field(
        field_name="alpha.gas", field_dict=field_dict, **kwargs
    )
    d_gas, field_dict = read_bubble_diameter(field_dict=field_dict, **kwargs)
    alpha_gas = _field_filter(alpha_gas, ind=ind_liq, field_type="scalar")
    d_gas = _field_filter(d_gas, ind=ind_liq, field_type="scalar")
    return (6.0 / d_gas) * alpha_gas, field_dict


def _instantaneous_kla_field(
    case_folder: str,
    time_folder: str,
    species_names: str | list[str],
    n_cells: int | None = None,
    field_dict: dict | None = None,
) -> tuple[dict, dict]:
    """Per-cell kLa = kL * a over the liquid, per species."""
    if field_dict is None:
        field_dict = {}
    if isinstance(species_names, str):
        species_names = [species_names]
    kl_spec_field, field_dict = _instantaneous_kl_field(
        case_folder, time_folder, species_names, n_cells, field_dict
    )
    a_field, field_dict = _instantaneous_a_field(
        case_folder, time_folder, n_cells, field_dict
    )
    kla_spec_field = {
        species_name: kl_spec_field[species_name] * a_field
        for species_name in species_names
    }
    return kla_spec_field, field_dict


def _instantaneous_cstar(
    case_folder: str,
    time_folder: str,
    species_names: str | list[str],
    n_cells: int | None = None,
    volume_time: str | None = None,
    field_dict: dict | None = None,
) -> tuple[dict, dict]:
    """Volume-averaged saturation concentration C* over the liquid, per species."""
    if field_dict is None:
        field_dict = {}
    if isinstance(species_names, str):
        species_names = [species_names]
    kwargs = {
        "case_folder": case_folder,
        "time_folder": time_folder,
        "n_cells": n_cells,
    }
    kwargs_vol = {
        "case_folder": case_folder,
        "time_folder": volume_time,
        "n_cells": n_cells,
    }
    globalVars = read_global_vars(case_folder=case_folder, cross_ref=True)
    mw_species = {}
    for species_name in species_names:
        if not f"He_{species_name}" in globalVars:
            err_msg = f"He_{species_name} was not found in globalVars."
            err_msg += f'\nIf you add it, it should be looking like #calc "$H_{species_name}_298 * exp($DH_{species_name} *(1. / $T0 - 1./298.15))";'
            raise KeyError(err_msg)
        mw_species[species_name] = species_name_to_mw(
            case_folder=case_folder, species_name=species_name
        )
    ind_liq, field_dict = _get_ind_liq(field_dict=field_dict, **kwargs)
    alpha_gas, field_dict = read_field(
        field_name="alpha.gas", field_dict=field_dict, **kwargs
    )
    rho_gas, field_dict = read_field(
        field_name="thermo:rho.gas", field_dict=field_dict, **kwargs
    )
    species_gas = {}
    for species_name in species_names:
        species_gas[species_name], field_dict = read_field(
            field_name=f"{species_name}.gas", field_dict=field_dict, **kwargs
        )
    alpha_gas = _field_filter(alpha_gas, ind=ind_liq, field_type="scalar")
    alpha_liq = 1 - alpha_gas
    rho_gas = _field_filter(rho_gas, ind=ind_liq, field_type="scalar")
    for species_name in species_names:
        species_gas[species_name] = _field_filter(
            species_gas[species_name], ind=ind_liq, field_type="scalar"
        )
    cell_volume, field_dict = read_cell_volumes(
        field_dict=field_dict, **kwargs_vol
    )
    cell_volume = _field_filter(cell_volume, ind=ind_liq, field_type="scalar")

    cstar_spec = {}
    for species_name in species_names:
        cstar_field = (
            rho_gas
            * species_gas[species_name]
            * globalVars[f"He_{species_name}"]
        ) / mw_species[species_name]
        cstar_spec[species_name] = _weighted_average(
            cstar_field, cell_volume * alpha_liq
        )
    return cstar_spec, field_dict


def compute_instantaneous_kla(
    case_folder: str,
    time_folder: str,
    species_names: str | list[str],
    n_cells: int | None = None,
    volume_time: str | None = None,
    field_dict: dict | None = None,
) -> tuple[dict, dict, dict]:
    r"""
    Calculate :math:`kLa_{\rm spec}` and saturation concentration (:math:`C^*_{\rm spec}`) for a list of species from instantaneous data (rather than doing a fit over time).

    :math:`kLa_{\rm spec}` for the species computed from Eq 7 and 8 in "Computational fluid dynamics study of full-scale aerobic bioreactors: Evaluation of gas–liquid mass transfer, oxygen uptake, and dynamic oxygen distribution", M. J. Rahimi, H. Sitaraman, D. Humbird, J. J. Stickel, Chem. Eng. Research and Design, Vol. 139, pp 293-295, 2018.



    .. math::

       \frac{1}{V_{\rm liq, tot}} \int_{V_{\rm liq}} kLa_{\rm spec} dV

    .. math::

       kLa_{\rm spec} = 3600 \sqrt{\frac{4 D_{\rm spec} |u_{\rm slip}|}{\pi d_{\rm gas}}} \frac{6 \alpha_{\rm gas}}{d_{\rm gas}}

    .. math::

       kLa_{\rm spec} = (\frac{2}{\pi^{1/2}} \times 3600) Re^{1/2} \frac{\mu_{\rm liq}^{1/2}}{D_{\rm spec}^{1/2} \rho_{\rm liq}^{1/2}} \frac{D_{\rm spec}}{d_{\rm gas}} \frac{6}{d_{\rm gas}} \alpha_{\rm gas}

    .. math::

       Re = \frac{\rho_{\rm liq} |u_{\rm slip}| d_{\rm gas}}{\mu_{\rm liq}}

    where:
      - :math:`kLa_{\rm spec}` is the mass transfer rate in :math:`h^{-1}`
      - :math:`d_{\rm gas}` is the bubble diameter in :math:`m`. Either read from the time folder, or looked up from phaseProperties
      - :math:`\alpha_{\rm gas}` is the volume fraction of gas. Read from the time folder.
      - :math:`\mu_{\rm liq}` is the liquid viscosity in :math:`kg.m^{-1}.s^{-1}`. Either read from the time folder or globalVars.
      - :math:`\rho_{\rm liq}` is the liquid density in :math:`kg.m^{-3}`. Either read from the time folder or assumed to be 1000kg/m3
      - :math:`D_{\rm spec}` is the species molecular diffusivity in :math:`m^2.s^{-1}`. Read from globalVars
      - :math:`|u_{\rm slip}|` is the magnitude of the slip velocity in :math:`m.s^{-1}`. Read from the time folder.
      - :math:`V_{\rm liq}` is the volume of liquid in :math:`m^3`. Read from the time folder.

     .. math::

       \frac{1}{V_{\rm liq, tot}} \int_{V_{\rm liq}} C^*_{\rm spec} dV

    :math:`C^*_{\rm spec}` computed from Eq 10 in "Computational fluid dynamics study of full-scale aerobic bioreactors: Evaluation of gas–liquid mass transfer, oxygen uptake, and dynamic oxygen distribution", M. J. Rahimi, H. Sitaraman, D. Humbird, J. J. Stickel, Chem. Eng. Research and Design, Vol. 139, pp 293-295, 2018.

     .. math::

       C^*_{\rm spec} = \rho_{\rm gas} Y_{\rm spec, gas} He_{\rm spec} / W_{\rm spec}

     and
      - :math:`C^{*}_{\rm spec}` is the saturation concentration of species spec in :math:`mol.m^{-3}`
      - :math:`\rho_{\rm gas}` is the density of the gas in :math:`kg.m^{-3}`. Read from the time folder.
      - :math:`Y_{\rm spec, gas}` is the mass fraction of species spec in the gas phase. Read from the time folder.
      - :math:`He_{\rm spec}` is the Henry's constant of species spec. Read from globalVars.
      - :math:`W_{\rm spec}` is the molar mass of species spec in :math:`kg.mol^{-1}`. Read from globalVars.


    Parameters
    ----------
    case_folder: str
        Path to case folder
    time_folder: str
        Name of time folder to analyze
    species_names: str | list[str]
        List of species name for which to compute kla
    n_cells : int | None
        Number of cells in the domain.
        If None, it will deduced from the field reading
    volume_time : str | None
        Time folder to read to get the cell volumes.
        If None, finds volume time automatically
    field_dict : dict
        Dictionary of fields used to avoid rereading the same fields to calculate different quantities

    Returns
    ----------
    kla_spec: dict
        Instantaneous volume averaged kLa for each species, in h^-1
        Keys are species names
        Values are the kLa values
    cstar_spec: dict
        Instantaneous volume averaged cstar for each species, in mol.m^-3
        Keys are species names
        Values are the cstar values
    field_dict : dict
        Dictionary of fields read
    """
    if field_dict is None:
        field_dict = {}
    if isinstance(species_names, str):
        species_names = [species_names]

    kwargs = {
        "case_folder": case_folder,
        "time_folder": time_folder,
        "n_cells": n_cells,
    }
    kwargs_vol = {
        "case_folder": case_folder,
        "time_folder": volume_time,
        "n_cells": n_cells,
    }

    kla_spec_field, field_dict = _instantaneous_kla_field(
        case_folder, time_folder, species_names, n_cells, field_dict
    )

    # Volume average over the liquid
    ind_liq, field_dict = _get_ind_liq(field_dict=field_dict, **kwargs)
    cell_volume, field_dict = read_cell_volumes(
        field_dict=field_dict, **kwargs_vol
    )
    cell_volume = _field_filter(cell_volume, ind=ind_liq, field_type="scalar")
    kla_spec = {
        species_name: _weighted_average(
            kla_spec_field[species_name], cell_volume
        )
        for species_name in species_names
    }
    cstar_spec, field_dict = _instantaneous_cstar(
        case_folder,
        time_folder,
        species_names,
        n_cells,
        volume_time,
        field_dict,
    )
    return kla_spec, cstar_spec, field_dict


def compute_instantaneous_kl(
    case_folder: str,
    time_folder: str,
    species_names: str | list[str],
    n_cells: int | None = None,
    volume_time: str | None = None,
    field_dict: dict | None = None,
) -> tuple[dict, dict]:
    r"""
    Calculate the mass-transfer coefficient (:math:`kL_{\rm spec}`) and saturation concentration (:math:`C^*_{\rm spec}`) for a list of species from instantaneous data (rather than doing a fit over time).

    :math:`kL_{\rm spec}` is the penetration-theory coefficient that
    :func:`compute_instantaneous_kla` multiplies by the interfacial area
    :math:`a = 6 \alpha_{\rm gas} / d_{\rm gas}` to form :math:`kLa_{\rm spec}`
    (i.e. :math:`kLa_{\rm spec} = kL_{\rm spec}\, a`), volume averaged over the liquid.

    .. math::

       \frac{1}{V_{\rm liq, tot}} \int_{V_{\rm liq}} kL_{\rm spec} dV

    .. math::

       kL_{\rm spec} = 3600 \sqrt{\frac{4 D_{\rm spec} |u_{\rm slip}|}{\pi d_{\rm gas}}}

    where:
      - :math:`kL_{\rm spec}` is the mass transfer coefficient in :math:`m.h^{-1}`
      - :math:`d_{\rm gas}` is the bubble diameter in :math:`m`. Either read from the time folder, or looked up from phaseProperties
      - :math:`D_{\rm spec}` is the species molecular diffusivity in :math:`m^2.s^{-1}`. Read from globalVars
      - :math:`|u_{\rm slip}|` is the magnitude of the slip velocity in :math:`m.s^{-1}`. Read from the time folder.
      - :math:`V_{\rm liq}` is the volume of liquid in :math:`m^3`. Read from the time folder.

    :math:`C^*_{\rm spec}` is computed as in :func:`compute_instantaneous_kla`.

    Parameters
    ----------
    case_folder: str
        Path to case folder
    time_folder: str
        Name of time folder to analyze
    species_names: str | list[str]
        List of species name for which to compute kL
    n_cells : int | None
        Number of cells in the domain.
        If None, it will deduced from the field reading
    volume_time : str | None
        Time folder to read to get the cell volumes.
        If None, finds volume time automatically
    field_dict : dict
        Dictionary of fields used to avoid rereading the same fields to calculate different quantities

    Returns
    ----------
    kl_spec: dict
        Instantaneous volume averaged kL for each species, in m.h^-1
        Keys are species names
        Values are the kL values
    cstar_spec: dict
        Instantaneous volume averaged cstar for each species, in mol.m^-3
        Keys are species names
        Values are the cstar values
    field_dict : dict
        Dictionary of fields read
    """
    if field_dict is None:
        field_dict = {}
    if isinstance(species_names, str):
        species_names = [species_names]

    kwargs = {
        "case_folder": case_folder,
        "time_folder": time_folder,
        "n_cells": n_cells,
    }
    kwargs_vol = {
        "case_folder": case_folder,
        "time_folder": volume_time,
        "n_cells": n_cells,
    }

    kl_spec_field, field_dict = _instantaneous_kl_field(
        case_folder, time_folder, species_names, n_cells, field_dict
    )

    # Volume average over the liquid
    ind_liq, field_dict = _get_ind_liq(field_dict=field_dict, **kwargs)
    cell_volume, field_dict = read_cell_volumes(
        field_dict=field_dict, **kwargs_vol
    )
    cell_volume = _field_filter(cell_volume, ind=ind_liq, field_type="scalar")
    kl_spec = {
        species_name: _weighted_average(
            kl_spec_field[species_name], cell_volume
        )
        for species_name in species_names
    }
    cstar_spec, field_dict = _instantaneous_cstar(
        case_folder,
        time_folder,
        species_names,
        n_cells,
        volume_time,
        field_dict,
    )
    return kl_spec, cstar_spec, field_dict


def compute_fitted_kla(
    case_folder: str,
    species_names: str | list[str],
    n_cells: int | None = None,
    volume_time: str | None = None,
    num_warmup: int = 4000,
    num_samples: int = 1000,
    field_dict: dict | None = None,
) -> tuple[dict, dict, dict]:
    r"""
    Calculate :math:`kLa_{\rm spec}` and saturation concentration (:math:`C^*_{\rm spec}`) for a list of species from time series data (rather than instantaneously).

    Given a time series of concentration of species, the following expression is fitted

    .. math::
       [spec](t) =  [spec]^* (1 - \operatorname{exp}(-{kLa}_{\rm spec} t)).

    where

      - :math:`kLa_{\rm spec}` is the mass transfer rate of species :math:`\rm spec` in :math:`h^{-1}`
      - :math:`t` is the time in :math:`s`
      - :math:`[spec]^*` is the estimated saturation concentration of species :math:`\rm spec` in :math:`mol/m^3`
      - :math:`[spec](t)` is the instantaneous liquid volume averaged concentration of species :math:`\rm spec` in :math:`mol/m^3`

    Both :math:`[spec]^*` and :math:`kLa_{\rm spec}` are fitted.
    The fit is done with Markov Chain Monte Carlo which outputs samples of the posterior PDF of :math:`[spec]^*` and :math:`kLa_{\rm spec}`.

    Parameters
    ----------
    case_folder: str
        Path to case folder
    species_names: str | list[str]
        List of species name for which to compute kla
    n_cells : int | None
        Number of cells in the domain.
        If None, it will deduced from the field reading
    volume_time : str | None
        Time folder to read to get the cell volumes.
        If None, finds volume time automatically
    num_warmup: int
        Number of MCMC samples in the warmup phase
        Defaults to 4000
    num_samples: int
        Number of posterior MCMC samples generated
        Defaults to 1000
    field_dict : dict
        Dictionary of fields used to avoid rereading the same fields to calculate different quantities

    Returns
    ----------
    kla_spec: dict
        Instantaneous volume averaged kLa for each species, in h^-1
        Keys are species names
        Values are dictionaries with key 'mean' (mean kLa value) and 'std' (1 standard deviation for the kLa value)
    cstar_spec: dict
        Instantaneous volume averaged cstar for each species, in mol.m^-3
        Keys are species names
        Values are dictionaries with key 'mean' (mean cstar value) and 'std' (1 standard deviation for the cstar value)
    field_dict : dict
        Dictionary of fields read
    """
    if field_dict is None:
        field_dict = {}

    if isinstance(species_names, str):
        species_names = [species_names]

    # Read relevant fields
    kwargs = {
        "case_folder": case_folder,
        "n_cells": n_cells,
        "volume_time": volume_time,
    }

    # Get all the time folders
    time_float_sorted, time_str_sorted = get_case_times(case_folder)

    # Get Mw of the species
    mw_species = {}
    for species_name in species_names:
        mw_species[species_name] = species_name_to_mw(
            case_folder=case_folder, species_name=species_name
        )

    # Initialize the mesh fields
    mesh_field_dict = {}
    if "cell_centers" in field_dict:
        mesh_field_dict["cell_centers"] = field_dict["cell_centers"]
    else:
        mesh_field_dict["cell_centers"], _ = read_cell_centers(case_folder)
    if "V" in field_dict:
        mesh_field_dict["V"] = field_dict["V"]
    else:
        mesh_field_dict["V"], _ = read_cell_volumes(case_folder)

    logger.info("Reading the species concentration history")

    # Initialize the data structure for concentration
    c_history = {}
    for species_name in species_names:
        c_history[species_name] = np.zeros(len(time_str_sorted))

    # Read concentration
    for itime, time_folder in enumerate(time_str_sorted):
        logger.debug(f"Reading {time_folder}")
        # Reinitialize kla field dict
        kla_field_dict = {}
        for key in mesh_field_dict:
            kla_field_dict[key] = mesh_field_dict[key]

        # Compute reactor averaged liquid concentration for all the species
        for species_name in species_names:
            c_liq, kla_field_dict = compute_ave_conc_liq(
                time_folder=time_folder,
                species_name=species_name,
                field_dict=kla_field_dict,
                **kwargs,
            )
        c_history[species_name][itime] = c_liq

    logger.info("Doing kla fit")
    # Compute kla
    kla_spec = {}
    cstar_spec = {}
    for species_name in species_names:
        kla_res = compute_kla(
            np.array(time_float_sorted),
            c_history[species_name],
            num_warmup=num_warmup,
            num_samples=num_samples,
        )
        # Convert to h-1
        kla_spec[species_name] = {
            "mean": kla_res["kla"] * 3600,
            "std": kla_res["kla_err"] * 3600,
        }
        cstar_spec[species_name] = {
            "mean": kla_res["cstar"],
            "std": kla_res["cstar_err"],
        }

    return kla_spec, cstar_spec, field_dict


def compute_fitted_kl(
    case_folder: str,
    species_names: str | list[str],
    n_cells: int | None = None,
    volume_time: str | None = None,
    num_warmup: int = 4000,
    num_samples: int = 1000,
    field_dict: dict | None = None,
) -> tuple[dict, dict, dict]:
    r"""Fitted mass-transfer coefficient :math:`kL = kLa / a`.

    Same as :func:`compute_fitted_kla` but the fitted mean and std of each
    species are divided by the interfacial area :math:`a = 6\,\epsilon / d`,
    evaluated at the last time folder.

    :param case_folder: path to the case folder
    :param species_names: species name(s) for which to compute kL
    :param n_cells: number of cells (deduced from the field read if None)
    :param volume_time: time folder for the cell volumes (auto if None)
    :param num_warmup: number of MCMC warmup samples
    :param num_samples: number of posterior MCMC samples
    :param field_dict: cache of already-read fields
    :return: ``(kl_spec, cstar_spec, field_dict)`` with kL mean/std in :math:`m.h^{-1}` and cstar mean/std in :math:`mol.m^{-3}`
    """
    if field_dict is None:
        field_dict = {}

    kla_spec, cstar_spec, field_dict = compute_fitted_kla(
        case_folder,
        species_names,
        n_cells=n_cells,
        volume_time=volume_time,
        num_warmup=num_warmup,
        num_samples=num_samples,
        field_dict=field_dict,
    )
    # interfacial area at the last time folder
    _, time_str_sorted = get_case_times(case_folder)
    last_time = time_str_sorted[-1]
    area_cache: dict = {}
    gas_holdup, area_cache = compute_gas_holdup(
        case_folder, last_time, n_cells, volume_time, area_cache
    )
    bubble_diam, area_cache = compute_ave_bubble_diam(
        case_folder, last_time, n_cells, volume_time, area_cache
    )
    area = interfacial_area(gas_holdup, bubble_diam)

    kl_spec = {
        species: {"mean": kla["mean"] / area, "std": kla["std"] / area}
        for species, kla in kla_spec.items()
    }
    return kl_spec, cstar_spec, field_dict
