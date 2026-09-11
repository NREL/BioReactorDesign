import numpy as np

from bird.utilities.ofio import (
    read_global_vars,
    read_gravity,
    read_surface_tension,
)

from .kla import compute_instantaneous_kl
from .phase import compute_ave_liquid_density, compute_ave_liquid_velocity


def froude(velocity: float, length: float, gravity: float = 9.81) -> float:
    r"""Froude number :math:`Fr = U / \sqrt{g\,L}`.

    :param velocity: characteristic velocity :math:`U` [m/s]
    :param length: characteristic length :math:`L` [m]
    :param gravity: gravitational acceleration :math:`g` [m/s2]
    :return: Froude number [-]
    """
    return velocity / np.sqrt(gravity * length)


def weber(
    density: float, velocity: float, length: float, surface_tension: float
) -> float:
    r"""Weber number :math:`We = \rho\,U^2\,L / \sigma`.

    :param density: fluid density :math:`\rho` [kg/m3]
    :param velocity: characteristic velocity :math:`U` [m/s]
    :param length: characteristic length :math:`L` [m]
    :param surface_tension: surface tension :math:`\sigma` [N/m]
    :return: Weber number [-]
    """
    return density * velocity**2 * length / surface_tension


def sherwood(kl: float, length: float, diffusivity: float) -> float:
    r"""Sherwood number :math:`Sh = k_L\,L / D`.

    :param kl: mass-transfer coefficient :math:`k_L` [m/s]
    :param length: characteristic length :math:`L` [m]
    :param diffusivity: mass diffusivity :math:`D` [m2/s]
    :return: Sherwood number [-]
    """
    return kl * length / diffusivity


def compute_froude_number(
    case_folder: str,
    time_folder: str,
    length: float,
    n_cells: int | None = None,
    volume_time: str | None = None,
    field_dict: dict | None = None,
) -> tuple[float, dict]:
    r"""Froude number from the liquid-averaged velocity and a given length.

    :math:`Fr = U / \sqrt{g L}` with :math:`U` the liquid-volume-averaged
    :math:`|U_{\rm liq}|` at ``time_folder``, :math:`g` from ``constant/g``, and
    :math:`L` the passed length.

    :param case_folder: path to the case folder
    :param time_folder: name of the time folder to analyze
    :param length: characteristic length :math:`L` [m]
    :param n_cells: number of cells (deduced from the field read if None)
    :param volume_time: time folder for the cell volumes (auto if None)
    :param field_dict: cache of already-read fields
    :return: ``(froude_number, field_dict)``
    """
    if field_dict is None:
        field_dict = {}
    velocity, field_dict = compute_ave_liquid_velocity(
        case_folder, time_folder, n_cells, volume_time, field_dict
    )
    gravity = read_gravity(case_folder)
    return froude(velocity, length, gravity=gravity), field_dict


def compute_weber_number(
    case_folder: str,
    time_folder: str,
    length: float,
    n_cells: int | None = None,
    volume_time: str | None = None,
    field_dict: dict | None = None,
) -> tuple[float, dict]:
    r"""Weber number from the liquid-averaged velocity/density and a given length.

    :math:`We = \rho U^2 L / \sigma` with :math:`U` and :math:`\rho` the
    liquid-volume-averaged velocity magnitude and density at ``time_folder``,
    :math:`\sigma` from ``constant/phaseProperties``, and :math:`L` the passed
    length.

    :param case_folder: path to the case folder
    :param time_folder: name of the time folder to analyze
    :param length: characteristic length :math:`L` [m]
    :param n_cells: number of cells (deduced from the field read if None)
    :param volume_time: time folder for the cell volumes (auto if None)
    :param field_dict: cache of already-read fields
    :return: ``(weber_number, field_dict)``
    """
    if field_dict is None:
        field_dict = {}
    velocity, field_dict = compute_ave_liquid_velocity(
        case_folder, time_folder, n_cells, volume_time, field_dict
    )
    density, field_dict = compute_ave_liquid_density(
        case_folder, time_folder, n_cells, volume_time, field_dict
    )
    surface_tension = read_surface_tension(case_folder)
    return weber(density, velocity, length, surface_tension), field_dict


def compute_sherwood_number(
    case_folder: str,
    time_folder: str,
    length: float,
    species_name: str,
    n_cells: int | None = None,
    volume_time: str | None = None,
    field_dict: dict | None = None,
) -> tuple[float, dict]:
    r"""Sherwood number for one species from the instantaneous kL.

    :math:`Sh = k_L L / D` with :math:`k_L` from
    :func:`compute_instantaneous_kl` at ``time_folder`` and the molecular
    diffusivity :math:`D` read as ``D_<species>`` from globalVars (the standard
    Sherwood definition; the turbulent contribution is deliberately excluded).

    :param case_folder: path to the case folder
    :param time_folder: name of the time folder to analyze
    :param length: characteristic length :math:`L` [m]
    :param species_name: species for which to compute kL and use D_<species>
    :param n_cells: number of cells (deduced from the field read if None)
    :param volume_time: time folder for the cell volumes (auto if None)
    :param field_dict: cache of already-read fields
    :return: ``(sherwood_number, field_dict)``
    """
    if field_dict is None:
        field_dict = {}
    kl_spec, _, field_dict = compute_instantaneous_kl(
        case_folder,
        time_folder,
        species_name,
        n_cells=n_cells,
        volume_time=volume_time,
        field_dict=field_dict,
    )
    diffusivity = float(
        read_global_vars(case_folder=case_folder, cross_ref=True)[
            f"D_{species_name}"
        ]
    )
    return sherwood(kl_spec[species_name], length, diffusivity), field_dict
