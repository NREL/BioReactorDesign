import numpy as np

from bird.utilities.ofio import *

logger = logging.getLogger(__name__)


def read_field(
    case_folder: str,
    time_folder: str,
    field_name: str,
    n_cells: int | None = None,
    field_dict: dict = {},
) -> tuple[np.ndarray | float, dict]:
    """
    Read field at a given time and store it in dictionary for later reuse

    Parameters
    ----------
    case_folder: str
        Path to case folder
    time_folder: str
        Name of time folder to analyze
    field_name: str
        Name of the field file to read
    n_cells : int | None
        Number of cells in the domain
    field_dict : dict
        Dictionary of fields used to avoid rereading the same fields to calculate different quantities

    Returns
    ----------
    field : np.ndarray | float
        Field read
    field_dict : dict
        Dictionary of fields read
    """

    if not (field_name in field_dict) or field_dict[field_name] is None:
        # Read field if it had not been read before
        field_file = os.path.join(case_folder, time_folder, field_name)
        field = readOF(field_file, n_cells=n_cells)["field"]
        field_dict[field_name] = field
    else:
        # Get field from dict if it has been read before
        field = field_dict[field_name]

    return field, field_dict


def read_cell_volume(
    case_folder: str,
    time_folder: str,
    n_cells: int | None = None,
    field_dict: dict = {},
) -> tuple[np.ndarray | float, dict]:
    """
    Read volume at a given time and store it in dictionary for later reuse

    Parameters
    ----------
    case_folder: str
        Path to case folder
    time_folder: str
        Name of time folder to analyze
    n_cells : int | None
        Number of cells in the domain
    field_dict : dict
        Dictionary of fields used to avoid rereading the same fields to calculate different quantities

    Returns
    ----------
    cell_volume : np.ndarray | float
        cell volume read from file
    field_dict : dict
        Dictionary of fields read
    """
    kwargs_vol = {
        "case_folder": case_folder,
        "time_folder": time_folder,
        "n_cells": n_cells,
    }
    try:
        cell_volume, field_dict = read_field(
            field_name="V", field_dict=field_dict, **kwargs_vol
        )
    except FileNotFoundError:
        error_msg = (
            f"Could not find {os.path.join(case_folder,volume_time,'V')}\n"
        )
        error_msg += "You can generate V with\n\t"
        error_msg += f"`postProcess -func writeCellVolumes -time {time_folder} -case {case_folder}`"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    return cell_volume, field_dict


def read_cell_centers(
    case_folder: str,
    cell_centers_file: str,
    field_dict: dict = {},
) -> tuple[np.ndarray, dict]:
    """
    Read volume at a given time and store it in dictionary for later reuse

    Parameters
    ----------
    case_folder: str
        Path to case folder
    cell_centers_file : str
        Filename of cell center data
    field_dict : dict
        Dictionary of fields used to avoid rereading the same fields to calculate different quantities

    Returns
    ----------
    cell_centers : np.ndarray
        cell centers read from file
    field_dict : dict
        Dictionary of fields read
    """

    if (
        not ("cell_centers" in field_dict)
        or field_dict["cell_centers"] is None
    ):
        try:
            cell_centers = readMesh(
                os.path.join(case_folder, cell_centers_file)
            )
            field_dict["cell_centers"] = cell_centers
        except FileNotFoundError:
            error_msg = f"Could not find {cell_centers_file}"
            error_msg += "You can generate it with\n\t"
            error_msg += f"`writeMeshObj -case {case_folder}`\n"
            time_float, time_str = getCaseTimes(case_folder)
            correct_ph = f"meshCellCentres_{time_str[0]}.obj"
            if not correct_path == cell_centers_file:
                error_msg += (
                    f"And adjust the cell center file path to {correct_path}"
                )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
    else:
        cell_centers = field_dict["cell_centers"]

    return cell_centers, field_dict


def get_ind_liq(
    case_folder: str | None = None,
    time_folder: str | None = None,
    n_cells: int | None = None,
    field_dict: dict = {},
) -> tuple[np.ndarray | float, dict]:
    """
    Get indices of pure liquid cells (where alpha.liquid > 0.5)

    Parameters
    ----------
    case_folder: str
        Path to case folder
    time_folder: str
        Name of time folder to analyze
    n_cells : int | None
        Number of cells in the domain
    field_name: str
        Name of the field file to read
    field_dict : dict
        Dictionary of fields used to avoid rereading the same fields to calculate different quantities

    Returns
    ----------
    ind_liq : np.ndarray | float
        indices of pure liquid cells
    field_dict : dict
        Dictionary of fields read
    """

    kwargs = {
        "case_folder": case_folder,
        "time_folder": time_folder,
        "n_cells": n_cells,
    }

    # Compute indices of pure liquid
    if not ("ind_liq" in field_dict) or field_dict["ind_liq"] is None:
        alpha_liq, field_dict = read_field(
            case_folder,
            time_folder,
            field_name="alpha.liquid",
            n_cells=n_cells,
            field_dict=field_dict,
        )
        ind_liq = np.argwhere(alpha_liq > 0.5)
        field_dict["ind_liq"] = ind_liq
    else:
        ind_liq = field_dict["ind_liq"]

    return ind_liq, field_dict


def get_ind_gas(
    case_folder: str | None = None,
    time_folder: str | None = None,
    n_cells: int | None = None,
    field_dict: dict = {},
) -> tuple[np.ndarray | float, dict]:
    """
    Get indices of pure gas cells (where alpha.liquid <= 0.5)

    Parameters
    ----------
    case_folder: str
        Path to case folder
    time_folder: str
        Name of time folder to analyze
    n_cells : int | None
        Number of cells in the domain
    field_name: str
        Name of the field file to read
    field_dict : dict
        Dictionary of fields used to avoid rereading the same fields to calculate different quantities

    Returns
    ----------
    ind_gas : np.ndarray | float
        indices of pure gas cells
    field_dict : dict
        Dictionary of fields read
    """

    kwargs = {
        "case_folder": case_folder,
        "time_folder": time_folder,
        "n_cells": n_cells,
    }

    # Compute indices of pure liquid
    if not ("ind_gas" in field_dict) or field_dict["ind_gas"] is None:
        alpha_liq = read_field(
            case_folder,
            time_folder,
            field_name="alpha.liquid",
            n_cells=n_cells,
            field_dict=field_dict,
        )
        ind_gas = np.argwhere(alpha_liq <= 0.5)
        field_dict["ind_gas"] = ind_gas
    else:
        ind_gas = field_dict["ind_gas"]

    return ind_gas, field_dict


def compute_gas_holdup(
    case_folder: str,
    time_folder: str,
    n_cells: int | None = None,
    volume_time: str = "0",
    field_dict: dict = {},
) -> tuple[float, dict]:
    """
    Calculate volume averaged gas hold up at a given time

    .. math::
       \\frac{1}{V_{\\rm liq, tot}} \int_{V_{\\rm liq}} (1-\\alpha_{\\rm liq}) dV

    where:
      - :math:`V_{\\rm liq, tot}` is the total volume of liquid
      - :math:`\\alpha_{\\rm liq}` is the liquid phase volume fraction
      - :math:`V` is the volume of the cells where :math:`\\alpha_{\\rm liq}` is measured

    Parameters
    ----------
    case_folder: str
        Path to case folder
    time_folder: str
        Name of time folder to analyze
    n_cells : int | None
        Number of cells in the domain
    volume_time : str
        Time folder to read to get the cell volumes
    field_dict : dict
        Dictionary of fields used to avoid rereading the same fields to calculate different quantities

    Returns
    ----------
    gas_holdup: float
        Volume averaged gas holdup
    field_dict : dict
        Dictionary of fields read
    """
    # Read relevant fields
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

    alpha_liq, field_dict = read_field(
        field_name="alpha.liquid", field_dict=field_dict, **kwargs
    )
    ind_liq, field_dict = get_ind_liq(field_dict=field_dict, **kwargs)
    cell_volume, field_dict = read_cell_volume(
        field_dict=field_dict, **kwargs_vol
    )

    # Only compute over the liquid
    if isinstance(alpha_liq, np.ndarray):
        alpha_liq = alpha_liq[ind_liq]
    else:
        raise TypeError
    if isinstance(cell_volume, np.ndarray):
        cell_volume = cell_volume[ind_liq]
    else:
        raise TypeError

    # Calculate
    gas_holdup = np.sum((1 - alpha_liq) * cell_volume) / np.sum(cell_volume)

    return gas_holdup, field_dict


def compute_superficial_velocity(
    case_folder: str,
    time_folder: str,
    n_cells: int | None = None,
    volume_time: str = "0",
    direction: int = 1,
    cell_centers_file: str = "meshCellCentres_0.obj",
    field_dict: dict = {},
) -> tuple[float, dict]:
    """
    Calculate superficial velocity (in m/s) in a given direction at a given time

    Parameters
    ----------
    case_folder: str
        Path to case folder
    time_folder: str
        Name of time folder to analyze
    n_cells : int | None
        Number of cells in the domain
    volume_time : str
        Time folder to read to get the cell volumes
    direction :  int
        Direction along which to calculate the superficial velocity
    cell_centers_file : str
        Filename of cell center data
    field_dict : dict
        Dictionary of fields used to avoid rereading the same fields to calculate different quantities

    Returns
    ----------
    sup_vel: float
        Superficial velocity (in m/s)
    field_dict : dict
        Dictionary of fields read
    """

    # Read relevant fields
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
    alpha_gas, field_dict = read_field(
        field_name="alpha.gas", field_dict=field_dict, **kwargs
    )
    U_gas, field_dict = read_field(
        field_name="U.gas", field_dict=field_dict, **kwargs
    )
    U_gas = U_gas[:, direction]

    cell_volume, field_dict = read_cell_volume(
        field_dict=field_dict, **kwargs_vol
    )
    cell_centers, field_dict = read_cell_centers(
        case_folder=case_folder,
        cell_centers_file=cell_centers_file,
        field_dict=field_dict,
    )

    # Find all cells in the middle of the domain
    max_dir = np.amax(cell_centers[:, direction])
    min_dir = np.amin(cell_centers[:, direction])
    middle_dir = (max_dir + min_dir) / 2
    tol = (max_dir - min_dir) * 0.05
    ind_middle = np.argwhere(
        (cell_centers[:, direction] >= middle_dir - tol)
        & (cell_centers[:, direction] <= middle_dir + tol)
    )

    # Only compute in the middle
    if isinstance(alpha_gas, np.ndarray):
        alpha_gas = alpha_gas[ind_middle]
    else:
        raise TypeError
    if isinstance(cell_volume, np.ndarray):
        cell_volume = cell_volume[ind_middle]
    else:
        raise TypeError
    if isinstance(U_gas, np.ndarray):
        U_gas = U_gas[ind_middle]
    else:
        raise TypeError

    # Compute
    sup_vel = np.sum(U_gas * alpha_gas * cell_volume) / np.sum(cell_volume)

    return sup_vel, field_dict


def compute_ave_y_liq(
    case_folder: str,
    time_folder: str,
    n_cells: int | None = None,
    volume_time: str = "0",
    spec_name: str = "CO2",
    field_dict={},
) -> tuple[float, dict]:
    """
    Calculate liquid volume averaged mass fraction of a species at a given time

    .. math::
       \\frac{1}{V_{\\rm liq, tot}} \int_{V_{\\rm liq}} Y dV_{\\rm liq}

    where:
      - :math:`V_{\\rm liq, tot}` is the toal volume of liquid
      - :math:`Y` is the species mass fraction
      - :math:`V_{\\rm liq}` is the volume of liquid where :math:`D` is measured


    Parameters
    ----------
    case_folder: str
        Path to case folder
    time_folder: str
        Name of time folder to analyze
    n_cells : int | None
        Number of cells in the domain
    volume_time : str
        Time folder to read to get the cell volumes
    spec_name : str
        Name of the species
    field_dict : dict
        Dictionary of fields used to avoid rereading the same fields to calculate different quantities

    Returns
    ----------
    liq_ave_y: float
        Liquid volume averaged mass fraction
    field_dict : dict
        Dictionary of fields read
    """

    # Read relevant fields
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
    alpha_liq, field_dict = read_field(
        field_name="alpha.liquid", field_dict=field_dict, **kwargs
    )
    y_liq, field_dict = read_field(
        field_name=f"{spec_name}.liquid", field_dict=field_dict, **kwargs
    )
    ind_liq, field_dict = get_ind_liq(field_dict=field_dict, **kwargs)

    cell_volume, field_dict = read_cell_volume(
        field_dict=field_dict, **kwargs_vol
    )

    # Only compute over the liquid
    if isinstance(alpha_liq, np.ndarray):
        alpha_liq = alpha_liq[ind_liq]
    else:
        raise TypeError
    if isinstance(cell_volume, np.ndarray):
        cell_volume = cell_volume[ind_liq]
    else:
        raise TypeError
    if isinstance(y_liq, np.ndarray):
        y_liq = y_liq[ind_liq]
    else:
        raise TypeError

    # Calculate
    liq_ave_y = np.sum(alpha_liq * y_liq * cell_volume) / np.sum(
        alpha_liq * cell_volume
    )

    return liq_ave_y, field_dict


def compute_ave_conc_liq(
    case_folder: str,
    time_folder: str,
    n_cells: int | None = None,
    volume_time: str = "0",
    spec_name: str = "CO2",
    mol_weight: float = 0.04401,
    rho_val: float | None = 1000,
    field_dict={},
) -> tuple[float, dict]:
    """
    Calculate liquid volume averaged concentration of a species at a given time

    .. math::
       \\frac{1}{V_{\\rm liq, tot}} \int_{V_{\\rm liq}} \\rho_{\\rm liq} Y / W dV_{\\rm liq}

    where:
      - :math:`V_{\\rm liq, tot}` is the toal volume of liquid
      - :math:`\\rho_{\\rm liq}` is the liquid density
      - :math:`Y` is the species mass fraction
      - :math:`W` is the species molar mass
      - :math:`V_{\\rm liq}` is the volume of liquid where :math:`D` is measured

    Parameters
    ----------
    case_folder: str
        Path to case folder
    time_folder: str
        Name of time folder to analyze
    n_cells : int | None
        Number of cells in the domain
    volume_time : str
        Time folder to read to get the cell volumes
    spec_name : str
        Name of the species
    mol_weight : float
        Molecular weight of species (kg/mol)
    rho_val : float | None
        Constant density not available from time folder (kg/m3)
    field_dict : dict
        Dictionary of fields used to avoid rereading the same fields to calculate different quantities

    Returns
    ----------
    conc_ave: float
        Liquid volume averaged species concentration
    field_dict : dict
        Dictionary of fields read
    """

    logger.debug(
        f"Computing concentration for {spec_name} with molecular weight {mol_weight:.4g} kg/mol"
    )
    if rho_val is not None:
        logger.debug(f"Assuming liquid density {rho_val} kg/m3")

    # Read relevant fields
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
    alpha_liq, field_dict = read_field(
        field_name="alpha.liquid", field_dict=field_dict, **kwargs
    )
    y_liq, field_dict = read_field(
        field_name=f"{spec_name}.liquid", field_dict=field_dict, **kwargs
    )
    ind_liq, field_dict = get_ind_liq(field_dict=field_dict, **kwargs)

    cell_volume, field_dict = read_cell_volume(
        field_dict=field_dict, **kwargs_vol
    )

    # Density of liquid is not always printed (special case)
    if not ("rho_liq" in field_dict) or field_dict["rho_liq"] is None:
        if rho_val is not None:
            rho_liq = rho_val
            field_dict["rho_liq"] = rho_val
        else:
            rho_liq_file = os.path.join(case_folder, time_folder, "rhom")
            rho_liq = readOFScal(rho_liq_file, n_cells)["field"]
            field_dict["rho_liq"] = rho_liq
    else:
        rho_liq = field_dict["rho_liq"]

    # Only compute over the liquid
    if isinstance(y_liq, np.ndarray):
        y_liq = y_liq[ind_liq]
    else:
        raise TypeError
    if isinstance(alpha_liq, np.ndarray):
        alpha_liq = alpha_liq[ind_liq]
    else:
        raise TypeError
    if isinstance(alpha_liq, np.ndarray):
        cell_volume = cell_volume[ind_liq]
    else:
        raise TypeError
    if isinstance(rho_liq, np.ndarray):
        rho_liq = rho_liq[ind_liq]
    else:
        raise TypeError

    conc_loc = rho_liq * y_liq / mol_weight

    conc_ave = np.sum(conc_loc * alpha_liq * cell_volume) / np.sum(
        alpha_liq * cell_volume
    )

    return conc_ave, field_dict


def compute_ave_bubble_diam(
    case_folder: str,
    time_folder: str,
    n_cells: int | None = None,
    volume_time: str = "0",
    field_dict={},
) -> tuple[float, dict]:
    """
    Calculate averaged bubble diameter over the liquid volume

    .. math::

       \\frac{1}{V_{\\rm liq, tot}} \\int_{V_{\\rm liq}} `d_{\\rm gas}` dV

    where:
      - :math:`V_{\\rm liq, tot}` is the toal volume of liquid
      - :math:`d_{\\rm gas}` is the bubble diameter
      - :math:`V_{\\rm liq}` is the volume of liquid where :math:`d_{\\rm gas}` is measured


    Parameters
    ----------
    case_folder: str
        Path to case folder
    time_folder: str
        Name of time folder to analyze
    n_cells : int | None
        Number of cells in the domain
    volume_time : str
        Time folder to read to get the cell volumes
    field_dict : dict
        Dictionary of fields used to avoid rereading the same fields to calculate different quantities

    Returns
    ----------
    diam: float
        Volume averaged gas holdup
    field_dict : dict
        Dictionary of fields read
    """

    # Read relevant fields
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
    alpha_liq, field_dict = read_field(
        field_name="alpha.liquid", field_dict=field_dict, **kwargs
    )
    d_gas, field_dict = read_field(
        field_name="d.gas", field_dict=field_dict, **kwargs
    )
    ind_liq, field_dict = get_ind_liq(field_dict=field_dict, **kwargs)

    cell_volume, field_dict = read_cell_volume(
        field_dict=field_dict, **kwargs_vol
    )

    # Only compute over the liquid
    if isinstance(d_gas, np.ndarray):
        d_gas = d_gas[ind_liq]
    else:
        raise TypeError
    if isinstance(alpha_liq, np.ndarray):
        alpha_liq = alpha_liq[ind_liq]
    else:
        raise TypeError
    if isinstance(alpha_liq, np.ndarray):
        cell_volume = cell_volume[ind_liq]
    else:
        raise TypeError

    # Calculate
    diam = np.sum(d_gas * alpha_liq * cell_volume) / np.sum(
        alpha_liq * cell_volume
    )

    return diam, field_dict
