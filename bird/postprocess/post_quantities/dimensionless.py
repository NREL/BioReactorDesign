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

    Parameters
    ----------
    velocity: float
        Characteristic velocity :math:`U`, in :math:`m.s^{-1}`
    length: float
        Characteristic length :math:`L`, in :math:`m`
    gravity: float
        Gravitational acceleration :math:`g`, in :math:`m.s^{-2}`

    Returns
    ----------
    froude_number: float
        Froude number (dimensionless)
    """
    return velocity / np.sqrt(gravity * length)


def weber(
    density: float, velocity: float, length: float, surface_tension: float
) -> float:
    r"""Weber number :math:`We = \rho\,U^2\,L / \sigma`.

    Parameters
    ----------
    density: float
        Fluid density :math:`\rho`, in :math:`kg.m^{-3}`
    velocity: float
        Characteristic velocity :math:`U`, in :math:`m.s^{-1}`
    length: float
        Characteristic length :math:`L`, in :math:`m`
    surface_tension: float
        Surface tension :math:`\sigma`, in :math:`N.m^{-1}`

    Returns
    ----------
    weber_number: float
        Weber number (dimensionless)
    """
    return density * velocity**2 * length / surface_tension


def sherwood(kl: float, length: float, diffusivity: float) -> float:
    r"""Sherwood number :math:`Sh = k_L\,L / D`.

    Parameters
    ----------
    kl: float
        Mass-transfer coefficient :math:`k_L`, in :math:`m.s^{-1}`
    length: float
        Characteristic length :math:`L`, in :math:`m`
    diffusivity: float
        Mass diffusivity :math:`D`, in :math:`m^2.s^{-1}`

    Returns
    ----------
    sherwood_number: float
        Sherwood number (dimensionless)
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

    Parameters
    ----------
    case_folder: str
        Path to case folder
    time_folder: str
        Name of the time folder to analyze
    length: float
        Characteristic length :math:`L`, in :math:`m`
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
    froude_number: float
        Froude number (dimensionless)
    field_dict : dict
        Dictionary of fields read
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

    Parameters
    ----------
    case_folder: str
        Path to case folder
    time_folder: str
        Name of the time folder to analyze
    length: float
        Characteristic length :math:`L`, in :math:`m`
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
    weber_number: float
        Weber number (dimensionless)
    field_dict : dict
        Dictionary of fields read
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

    Parameters
    ----------
    case_folder: str
        Path to case folder
    time_folder: str
        Name of the time folder to analyze
    length: float
        Characteristic length :math:`L`, in :math:`m`
    species_name: str
        Species for which to compute kL and use D_<species>
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
    sherwood_number: float
        Sherwood number (dimensionless)
    field_dict : dict
        Dictionary of fields read
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
    # kL from compute_instantaneous_kl is in m/h, but D is in m2/s, so kL must
    # be converted back to m/s for Sh to come out dimensionless.
    kl_m_per_s = kl_spec[species_name] / 3600
    return sherwood(kl_m_per_s, length, diffusivity), field_dict
