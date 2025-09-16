import numpy as np

from bird import logger
from bird.utilities.ofio import *
from paraview import simple as pv
import vtk.numpy_interface.dataset_adapter as dsa



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
    field_dict: dict | None = None,
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
    field_dict : dict | None
        Dictionary of fields used to avoid rereading the same fields to calculate different quantities

    Returns
    ----------
    ind_liq : np.ndarray | float
        indices of pure liquid cells
    field_dict : dict
        Dictionary of fields read
    """
    if field_dict is None:
        return None

    kwargs = {
        "case_folder": case_folder,
        "time_folder": time_folder,
        "n_cells": n_cells,
    }

    logger.warning("Assuming that alpha_liq > 0.5 denotes pure liquid")

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


def _get_ind_gas(
    case_folder: str,
    time_folder: str,
    n_cells: int | None = None,
    field_dict: dict | None = None,
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
    field_dict : dict | None
        Dictionary of fields used to avoid rereading the same fields to calculate different quantities

    Returns
    ----------
    ind_gas : np.ndarray | float
        indices of pure gas cells
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

    logger.warning("Assuming that alpha_liq <= 0.5 denotes pure gas")

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


def _get_ind_slice(
    case_folder: str,
    location: float,
    direction: int | None = None,
    tolerance: float | None = None,
    cell_centers_file: str | None = None,
    field_dict: dict | None = None,
) -> tuple[np.ndarray | float, dict]:
    """
    Get indices of cells along a slice given by a direction and a location

    Parameters
    ----------
    case_folder: str
        Path to case folder
    location: float
        Axial location where to pick the cells
        If outside mesh bounds, will raise an error
    direction :  int
        Direction along which to calculate the superficial velocity.
        Must be in [0, 1, 2].
        If None, assume y direction
    tolerance : float
        Include cells where location is in [location - tolerance , location + tolerance].
        If None, it will be 2 times the axial mesh size
    cell_centers_file : str | None
        Filename of cell center data
        If None, finds cell center file automatically
    field_dict : dict
        Dictionary of fields used to avoid rereading the same fields to calculate different quantities

    Returns
    ----------
    ind_location : np.ndarray
        indices of cells along the desired slice
    field_dict : dict
        Dictionary of fields read
    """
    if field_dict is None:
        field_dict = {}

    if not (f"ind_location_{location:.2g}" in field_dict):

        cell_centers, field_dict = read_cell_centers(
            case_folder=case_folder,
            cell_centers_file=cell_centers_file,
            field_dict=field_dict,
        )

        if direction is None:
            logger.warning(
                "Assuming that axial direction is along the y direction"
            )
            direction = 1

        assert direction in [0, 1, 2]

        axial_cell_centers = np.sort(np.unique(cell_centers[:, direction]))
        if (
            location < axial_cell_centers.min()
            or location > axial_cell_centers.max()
        ):
            raise ValueError(
                f"Location {location:.2g} outside the mesh [{axial_cell_centers.min()}, {axial_cell_centers.max()}]"
            )

        if tolerance is None:
            ind_location_unique = np.argmin(abs(axial_cell_centers - location))
            if ind_location_unique == 0:
                tolerance = 2 * (axial_cell_centers[1] - axial_cell_centers[0])
            elif ind_location_unique == len(axial_cell_centers) - 1:
                tolerance = 2 * (
                    axial_cell_centers[-1] - axial_cell_centers[-2]
                )
            else:
                tolerance = (
                    axial_cell_centers[ind_location_unique + 1]
                    - axial_cell_centers[ind_location_unique - 1]
                )
            logger.debug(
                f"Tolerance for slice location not set, assuming {tolerance:.2g}"
            )

        # Do the actual filtering
        ind_location = np.argwhere(
            abs(cell_centers[:, direction] - location) <= tolerance
        )

        n_cells_location = len(ind_location)

        if n_cells_location == 0:
            raise ValueError(
                f"No cell found for location {location:.2g}, increase tolerance or check if location {location:.2g} is valid"
            )

        logger.debug(
            f"Found {n_cells_location} cells around location {location:.2g}"
        )
        field_dict[f"ind_location_{location:.2g}"] = ind_location

    else:
        ind_location = field_dict[f"ind_location_{location:.2g}"]

    return ind_location, field_dict


def compute_gas_holdup(
    case_folder: str,
    time_folder: str,
    n_cells: int | None = None,
    volume_time: str | None = None,
    field_dict: dict | None =None,
) -> tuple[float, dict]:
    r"""
    Calculate volume averaged gas hold up at a given time

    .. math::
       \frac{1}{V_{\rm liq, tot}} \int_{V_{\rm liq}} (1-\alpha_{\rm liq}) dV

    where:
      - :math:`V_{\rm liq, tot}` is the total volume of liquid
      - :math:`\alpha_{\rm liq}` is the liquid phase volume fraction
      - :math:`V` is the volume of the cells where :math:`\alpha_{\rm liq}` is measured

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
    field_dict : dict | None
        Dictionary of fields used to avoid rereading the same fields to calculate different quantities

    Returns
    ----------
    gas_holdup: float
        Volume averaged gas holdup
    field_dict : dict
        Dictionary of fields read
    """

    if field_dict is None:
        field_dict = {}

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
    cell_centers_file: str | None = None,
    height: float | None = None,
    use_pv: bool = False,
    field_dict: dict | None = None,
) -> tuple[float, dict]:
    r"""
    Calculate superficial gas velocity (in m/s) in a given direction at a given time

    .. math::
       \frac{1}{V_{\rm height, tot}} \int_{V_{\rm height}}  U_{\rm gas} \alpha_{\rm gas} dV

    where:
      - :math:`V_{\rm height, tot}` is the total volume of cells near the axial location considered
      - :math:`\alpha_{\rm gas}` is the gas phase volume fraction
      - :math:`U_{\rm gas}` is the gas phase velocity along the axial direction
      - :math:`V_{\rm height}` is the local volume of the cells where :math:`U_{\rm gas} \alpha_{\rm gas}` is measured (near the axial location considered)

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
    cell_centers_file : str | None
        Filename of cell center data
        If None, finds cell center file automatically
    height: float | None
        Axial location at which to compute the superficial velocity.
        If None, use the mid point of the liquid domain along the axial direction
    use_pv: bool
        Use paraview to create a slice in the middle of the reactor
        Default to False
    field_dict : dict | None
        Dictionary of fields used to avoid rereading the same fields to calculate different quantities

    Returns
    ----------
    sup_vel: float
        Superficial velocity (in m/s)
    field_dict : dict
        Dictionary of fields read
    """

    if field_dict is None:
        field_dict = {}

    # Read relevant fields
    kwargs = {
        "case_folder": case_folder,
        "time_folder": time_folder,
        "n_cells": n_cells,
    }
    if direction is None:
        logger.warning(
            "Assuming that superficial velocity is along the y direction"
        )
        direction = 1

    if direction not in [0, 1, 2]:
        raise ValueError(f"Direction ({direction}) must be in [0, 1, 2]")


    if use_pv:
        try:
            assert os.path.isdir(os.path.join(case_folder, "constant", "polyMesh"))
            assert os.path.isfile(os.path.join(case_folder, "constant", "polyMesh", "points"))
            assert os.path.isfile(os.path.join(case_folder, "constant", "polyMesh", "faces"))
            assert os.path.isfile(os.path.join(case_folder, "constant", "polyMesh", "owner"))
            assert os.path.isfile(os.path.join(case_folder, "constant", "polyMesh", "neighbour"))
            assert os.path.isfile(os.path.join(case_folder, "constant", "polyMesh", "boundary"))
        except AssertionError:
            logger.warning("Using ParaView requires to make a complete polyMesh, will not use ParaView")
            use_pv = False

    if use_pv:
        logger.debug("Using paraview for superficial velocity calculation")
        # Clean paraview pipelin
        for f in pv.GetSources().values():
            pv.Delete(f)

        # Set up openfoam case
        ofreader = pv.OpenFOAMReader(FileName = case_folder)
        ofreader.CaseType = 'Reconstructed Case'
        t = np.array(ofreader.TimestepValues)
        assert t.size > 0

        # Find the time to process
        time_pv_ind = np.argmin(abs(t-float(time_folder)))
        assert abs(t[time_pv_ind]-float(time_folder)) < 1e-12
        pv.UpdatePipeline(time = t[time_pv_ind])

        # Get liquid phase field
        logger.warning("Assuming that alpha_liq > 0.5 denotes pure liquid")
        liquidthreshold = pv.Threshold(Input=ofreader, Scalars=['CELLS', 'alpha.liquid'],\
                                       LowerThreshold=0.5,UpperThreshold=1.0,ThresholdMethod='Between')

        # Find extent of the liquid phase
        ofvtkdata = pv.servermanager.Fetch(liquidthreshold)
        ofdata = dsa.WrapDataObject( ofvtkdata)
        ofpts = np.array(ofdata.Points.Arrays[0])
        ptsmin_lt = ofpts.min(axis=0) # minimum values of the three axes
        ptsmax_lt = ofpts.max(axis=0) # maximum values of the three axes
       
        # Compute gas velocity in the liquid phase
        if direction == 0:
            u_gas_str = "U.gas_X"
        elif direction == 1:
            u_gas_str = "U.gas_Y"
        elif direction == 2:
            u_gas_str = "U.gas_Z"

        pv_calc  = pv.Calculator(Input=ofreader, AttributeType='Cell Data',\
                                 ResultArrayName='vflowrate',\
                                 Function=f'"alpha.gas"*"{u_gas_str}"')
        
        # create a new slice in the middle of the liquid domain
        slice_location = 0.5*(ptsmax_lt[direction]+ptsmin_lt[direction])
        pv_slice = pv.Slice(Input=pv_calc)

        origin = [0.0] * 3
        normal = [0.0] * 3
        origin[direction] = slice_location
        normal[direction] = 1.0
        
        pv_slice.SliceType.Origin = origin
        pv_slice.SliceType.Normal = normal
        
        # integrate variables along the slice
        pv_int = pv.IntegrateVariables(Input=pv_slice)
       
        # calculate superficial vel
        pv.UpdatePipeline(time=t[time_pv_ind])
        pv_dat = dsa.WrapDataObject( pv.servermanager.Fetch(pv_int) )
        vfrate  = pv_dat.CellData['vflowrate'].item()
        area = pv_dat.CellData['Area'].item()

        sup_vel = vfrate/area

    else:
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

        if U_gas.shape == (3,):
            # Uniform field
            U_gas_axial = U_gas[direction]
        else:
            # Non-uniform field
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

        ind_middle, field_dict = _get_ind_slice(
            case_folder=case_folder,
            location=height,
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
    field_dict: dict|None=None,
) -> tuple[float, dict]:
    r"""
    Calculate liquid volume averaged mass fraction of a species at a given time

    .. math::
       \frac{1}{V_{\rm liq, tot}} \int_{V_{\rm liq}} Y dV_{\rm liq}

    where:
      - :math:`V_{\rm liq, tot}` is the toal volume of liquid
      - :math:`Y` is the species mass fraction
      - :math:`V_{\rm liq}` is the volume of liquid where :math:`Y` is measured


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
    field_dict : dict | None
        Dictionary of fields used to avoid rereading the same fields to calculate different quantities

    Returns
    ----------
    liq_ave_y: float
        Liquid volume averaged mass fraction
    field_dict : dict
        Dictionary of fields read
    """
    if field_dict is None:
        field_dict = {}

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
    field_dict:dict|None=None,
) -> tuple[float, dict]:
    r"""
    Calculate liquid volume averaged concentration of a species at a given time

    .. math::
       \frac{1}{V_{\rm liq, tot}} \int_{V_{\rm liq}} \rho_{\rm liq} Y / W dV_{\rm liq}

    where:
      - :math:`V_{\rm liq, tot}` is the toal volume of liquid
      - :math:`\rho_{\rm liq}` is the liquid density
      - :math:`Y` is the species mass fraction
      - :math:`W` is the species molar mass
      - :math:`V_{\rm liq}` is the volume of liquid where :math:`Y` is measured

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
    if field_dict is None:
        field_dict = {}

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
    alpha_liq, field_dict = read_field(
        field_name="alpha.liquid", field_dict=field_dict, **kwargs
    )
    y_liq, field_dict = read_field(
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
            rho_liq = _readOFScal(rho_liq_file, n_cells)["field"]
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
    field_dict:dict|None=None,
) -> tuple[float, dict]:
    r"""
    Calculate averaged bubble diameter over the liquid volume

    .. math::

       \frac{1}{V_{\rm liq, tot}} \int_{V_{\rm liq}} d_{\rm gas} dV

    where:
      - :math:`V_{\rm liq, tot}` is the toal volume of liquid
      - :math:`d_{\rm gas}` is the bubble diameter
      - :math:`V_{\rm liq}` is the volume of liquid where :math:`d_{\rm gas}` is measured


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
    if field_dict is None:
        field_dict = {}

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
