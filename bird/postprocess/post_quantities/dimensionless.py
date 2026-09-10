import numpy as np

from bird.utilities.ofio import read_gravity


def froude(
    velocity: float, length: float, case_folder: str | None = None
) -> float:
    r"""Froude number :math:`Fr = U / \sqrt{g\,L}`.

    :param velocity: characteristic velocity :math:`U` [m/s]
    :param length: characteristic length :math:`L` [m]
    :param case_folder: case whose ``constant/g`` sets :math:`g`; 9.81 if None
        or the file is absent
    :return: Froude number [-]
    """
    gravity = read_gravity(case_folder) if case_folder is not None else 9.81
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
