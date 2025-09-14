import numpy as np

from bird.utilities.ofio import *

logger = logging.getLogger(__name__)


def _read_field(
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
        Number of cells in the domain.
        If None, it will deduced from the field reading
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


def _field_filter(
    field: float | np.ndarray, ind: np.ndarray, field_type: str
) -> float | np.ndarray:
    """
    Filter field by index. Handle uniform and non uniform fields

    Parameters
    ----------
    field: float | np.ndarray
        Field to filter
    ind: np.ndarray
        Cell indices to keep
    field_type : str
        Type of the field ("scalar" or "vector")

    Returns
    ----------
    filtered_field: float | np.ndarray
        Field filtered by cell indices

    """
    if field_type.lower() == "scalar":
        if isinstance(field, np.ndarray):
            if len(field.shape) > 1:
                err_msg = f"Scalar field shape {field.shape} but expected a flat array"
                raise ValueError(err_msg)
            filtered_field = field[ind]
        elif isinstance(field, float):
            # Uniform field
            filtered_field = field
        else:
            err_msg = f"Got field type {type(field)}."
            err_msg += " Expected float of np.ndarray for scalar field"
            raise TypeError(err_msg)

    elif field_type.lower() == "vector":
        if isinstance(field, np.ndarray):
            if field.shape == (3,):
                # Uniform field
                filtered_field = field
            else:
                filtered_field = field[ind]
        else:
            err_msg = f"Got field type {type(field)}."
            err_msg += " Expected np.ndarray for vector field"
            raise TypeError(err_msg)

    else:
        msg = f"Field type ({field_type}) not recognized"
        msg += " Supported field types are 'scalar' and 'vector'"
        raise NotImplementedError(msg)

    return filtered_field


def _get_ind_liq(
    case_folder: str,
    time_folder: str,
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
        Number of cells in the domain.
        If None, it will deduced from the field reading
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

    logger.warning("Assuming that alpha_liq > 0.5 denotes pure liquid")

    # Compute indices of pure liquid
    if not ("ind_liq" in field_dict) or field_dict["ind_liq"] is None:
        alpha_liq, field_dict = _read_field(
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


def _get_ind_gas(
    case_folder: str,
    time_folder: str,
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
        Number of cells in the domain.
        If None, it will deduced from the field reading
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

    logger.warning("Assuming that alpha_liq <= 0.5 denotes pure gas")

    # Compute indices of pure liquid
    if not ("ind_gas" in field_dict) or field_dict["ind_gas"] is None:
        alpha_liq = _read_field(
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


def _get_ind_height(
    height: float,
    case_folder: str,
    direction: int | None = None,
    tolerance: float | None = None,
    field_dict: dict = {},
) -> tuple[np.ndarray | float, dict]:
    """
    Get indices of pure gas cells (where alpha.liquid <= 0.5)

    Parameters
    ----------
    height: float
        Axial location where to pick the cells
    case_folder: str
        Path to case folder
    direction :  int | None
        Direction along which to calculate the superficial velocity.
        If None, assume y direction
    tolerance : float
        Include cells where height is in [height - tolerance , height + tolerance].
        If None, it will be 2 times the axial mesh size
    field_dict : dict
        Dictionary of fields used to avoid rereading the same fields to calculate different quantities

    Returns
    ----------
    ind_height : np.ndarray
        indices of cells at the desired height
    field_dict : dict
        Dictionary of fields read
    """

    if not (f"ind_height_{height:.2g}" in field_dict):

        cell_centers, field_dict = read_cell_centers(
            case_folder=case_folder,
            cell_centers_file=None,
            field_dict=field_dict,
        )

        if direction is None:
            logger.warning(
                "Assuming that axial direction is along the y direction"
            )
            direction = 1

        axial_cell_centers = np.sort(np.unique(cell_centers[:, direction]))
        if (
            height < axial_cell_centers.min()
            or height > axial_cell_centers.max()
        ):
            raise ValueError(
                f"Height {height:.2g} outside the mesh [{axial_cell_centers.min()}, {axial_cell_centers.max()}]"
            )

        if tolerance is None:
            ind_height_unique = np.argmin(abs(axial_cell_centers - height))
            if ind_height_unique == 0:
                tolerance = 2 * (axial_cell_centers[1] - axial_cell_centers[0])
            elif ind_height_unique == len(axial_cell_centers) - 1:
                tolerance = 2 * (
                    axial_cell_centers[-1] - axial_cell_centers[-2]
                )
            else:
                tolerance = (
                    axial_cell_centers[ind_height_unique + 1]
                    - axial_cell_centers[ind_height_unique - 1]
                )
            logger.warning(
                f"Tolerance for conditional height not set, assuming {tolerance:.2g}"
            )

        # Do the actual filtering
        ind_height = np.argwhere(
            abs(cell_centers[:, direction] - height) <= tolerance
        )

        n_cells_height = len(ind_height)

        if n_cells_height == 0:
            raise ValueError(
                f"No cell found for height {height:.2g}, increase tolerance or check if height {height:.2g} is valid"
            )

        logger.debug(
            f"Found {n_cells_height} cells around height {height:.2g}"
        )
        field_dict[f"ind_height_{height:.2g}"] = ind_height

    else:
        ind_height = field_dict[f"ind_height_{height:.2g}"]

    return ind_height, field_dict


def compute_gas_holdup(
    case_folder: str,
    time_folder: str,
    n_cells: int | None = None,
    volume_time: str | None = None,
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
        Number of cells in the domain.
        If None, it will deduced from the field reading
    volume_time : str | None
        Time folder to read to get the cell volumes.
        If None, finds volume time automatically
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

    alpha_liq, field_dict = _read_field(
        field_name="alpha.liquid", field_dict=field_dict, **kwargs
    )
    ind_liq, field_dict = _get_ind_liq(field_dict=field_dict, **kwargs)
    cell_volume, field_dict = read_cell_volumes(
        field_dict=field_dict, **kwargs_vol
    )

    # Only compute over the pure liquid
    alpha_liq = _field_filter(alpha_liq, ind=ind_liq, field_type="scalar")
    cell_volume = _field_filter(cell_volume, ind=ind_liq, field_type="scalar")

    # Calculate
    gas_holdup = np.sum((1 - alpha_liq) * cell_volume) / np.sum(cell_volume)

    return gas_holdup, field_dict


def compute_superficial_gas_velocity(
    case_folder: str,
    time_folder: str,
    n_cells: int | None = None,
    volume_time: str | None = None,
    direction: int | None = None,
    cell_centers_file: str = "meshCellCentres_0.obj",
    height: float | None = None,
    field_dict: dict = {},
) -> tuple[float, dict]:
    """
    Calculate superficial gas velocity (in m/s) in a given direction at a given time

    .. math::
       \\frac{1}{V_{\\rm height, tot}} \int_{V_{\\rm height}}  U_{\\rm gas} \\alpha_{\\rm gas} dV

    where:
      - :math:`V_{\\rm height, tot}` is the total volume of cells near the axial location considered
      - :math:`\\alpha_{\\rm gas}` is the gas phase volume fraction
      - :math:`U_{\\rm gas}` is the gas phase velocity along the axial direction
      - :math:`V_{\\rm height}` is the local volume of the cells where :math:`U_{\\rm gas} \\alpha_{\\rm gas})` is measured (near the axial location considered)

    Parameters
    ----------
    case_folder: str
        Path to case folder
    time_folder: str
        Name of time folder to analyze
    n_cells : int | None
        Number of cells in the domain.
        If None, it will deduced from the field reading
    volume_time : str | None
        Time folder to read to get the cell volumes.
        If None, finds volume time automatically
    direction :  int | None
        Direction along which to calculate the superficial velocity.
        If None, assume y direction
    cell_centers_file : str
        Filename of cell center data
    height: float | None
        Axial location at which to compute the superficial velocity.
        If None, use the mid point of the liquid domain along the axial direction
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
    alpha_gas, field_dict = _read_field(
        field_name="alpha.gas", field_dict=field_dict, **kwargs
    )
    U_gas, field_dict = _read_field(
        field_name="U.gas", field_dict=field_dict, **kwargs
    )

    if direction is None:
        logger.warning(
            "Assuming that superficial velocity is along the y direction"
        )
        direction = 1

    U_gas_axial = U_gas[:, direction]

    cell_volume, field_dict = read_cell_volumes(
        field_dict=field_dict, **kwargs_vol
    )
    cell_centers, field_dict = read_cell_centers(
        case_folder=case_folder,
        cell_centers_file=cell_centers_file,
        field_dict=field_dict,
    )

    if height is None:
        # Find all cells in the middle of the liquid domain
        ind_liq, field_dict = _get_ind_liq(field_dict=field_dict, **kwargs)
        max_dir = np.amax(cell_centers[ind_liq, direction])
        min_dir = np.amin(cell_centers[ind_liq, direction])
        height = (max_dir + min_dir) / 2

    ind_middle, field_dict = _get_ind_height(
        height=height,
        case_folder=case_folder,
        direction=direction,
        field_dict=field_dict,
    )

    # Filter fields to the right location
    alpha_gas = _field_filter(alpha_gas, ind=ind_middle, field_type="scalar")
    cell_volume = _field_filter(
        cell_volume, ind=ind_middle, field_type="scalar"
    )
    U_gas_axial = _field_filter(
        U_gas_axial, ind=ind_middle, field_type="scalar"
    )

    # Compute
    sup_vel = np.sum(U_gas_axial * alpha_gas * cell_volume) / np.sum(
        cell_volume
    )

    return sup_vel, field_dict


def compute_ave_y_liq(
    case_folder: str,
    time_folder: str,
    n_cells: int | None = None,
    volume_time: str | None = None,
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
      - :math:`V_{\\rm liq}` is the volume of liquid where :math:`Y` is measured


    Parameters
    ----------
    case_folder: str
        Path to case folder
    time_folder: str
        Name of time folder to analyze
    n_cells : int | None
        Number of cells in the domain.
        If None, it will deduced from the field reading
    volume_time : str | None
        Time folder to read to get the cell volumes.
        If None, finds volume time automatically
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
    alpha_liq, field_dict = _read_field(
        field_name="alpha.liquid", field_dict=field_dict, **kwargs
    )
    y_liq, field_dict = _read_field(
        field_name=f"{spec_name}.liquid", field_dict=field_dict, **kwargs
    )
    ind_liq, field_dict = _get_ind_liq(field_dict=field_dict, **kwargs)

    cell_volume, field_dict = read_cell_volumes(
        field_dict=field_dict, **kwargs_vol
    )

    # Only compute over the liquid
    alpha_liq = _field_filter(alpha_liq, ind=ind_liq, field_type="scalar")
    cell_volume = _field_filter(cell_volume, ind=ind_liq, field_type="scalar")
    y_liq = _field_filter(y_liq, ind=ind_liq, field_type="scalar")

    # Calculate
    liq_ave_y = np.sum(alpha_liq * y_liq * cell_volume) / np.sum(
        alpha_liq * cell_volume
    )

    return liq_ave_y, field_dict


def compute_ave_conc_liq(
    case_folder: str,
    time_folder: str,
    n_cells: int | None = None,
    volume_time: str | None = None,
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
      - :math:`V_{\\rm liq}` is the volume of liquid where :math:`Y` is measured

    Parameters
    ----------
    case_folder: str
        Path to case folder
    time_folder: str
        Name of time folder to analyze
    n_cells : int | None
        Number of cells in the domain.
        If None, it will deduced from the field reading
    volume_time : str | None
        Time folder to read to get the cell volumes.
        If None, finds volume time automatically
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
        rho_val = float(rho_val)
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
    alpha_liq, field_dict = _read_field(
        field_name="alpha.liquid", field_dict=field_dict, **kwargs
    )
    y_liq, field_dict = _read_field(
        field_name=f"{spec_name}.liquid", field_dict=field_dict, **kwargs
    )
    ind_liq, field_dict = _get_ind_liq(field_dict=field_dict, **kwargs)

    cell_volume, field_dict = read_cell_volumes(
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
    alpha_liq = _field_filter(alpha_liq, ind=ind_liq, field_type="scalar")
    cell_volume = _field_filter(cell_volume, ind=ind_liq, field_type="scalar")
    y_liq = _field_filter(y_liq, ind=ind_liq, field_type="scalar")
    rho_liq = _field_filter(rho_liq, ind=ind_liq, field_type="scalar")

    conc_loc = rho_liq * y_liq / mol_weight

    conc_ave = np.sum(conc_loc * alpha_liq * cell_volume) / np.sum(
        alpha_liq * cell_volume
    )

    return conc_ave, field_dict


def compute_ave_bubble_diam(
    case_folder: str,
    time_folder: str,
    n_cells: int | None = None,
    volume_time: str | None = None,
    field_dict={},
) -> tuple[float, dict]:
    """
    Calculate averaged bubble diameter over the liquid volume

    .. math::

       \\frac{1}{V_{\\rm liq, tot}} \\int_{V_{\\rm liq}} d_{\\rm gas} dV

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
        Number of cells in the domain.
        If None, it will deduced from the field reading
    volume_time : str | None
        Time folder to read to get the cell volumes.
        If None, finds volume time automatically
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
    alpha_liq, field_dict = _read_field(
        field_name="alpha.liquid", field_dict=field_dict, **kwargs
    )
    d_gas, field_dict = _read_field(
        field_name="d.gas", field_dict=field_dict, **kwargs
    )
    ind_liq, field_dict = _get_ind_liq(field_dict=field_dict, **kwargs)

    cell_volume, field_dict = read_cell_volumes(
        field_dict=field_dict, **kwargs_vol
    )

    # Only compute over the liquid
    alpha_liq = _field_filter(alpha_liq, ind=ind_liq, field_type="scalar")
    cell_volume = _field_filter(cell_volume, ind=ind_liq, field_type="scalar")
    d_gas = _field_filter(d_gas, ind=ind_liq, field_type="scalar")

    # Calculate
    diam = np.sum(d_gas * alpha_liq * cell_volume) / np.sum(
        alpha_liq * cell_volume
    )

    return diam, field_dict
