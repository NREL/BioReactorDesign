import logging

import numpy as np

logger = logging.getLogger(__name__)


def make_walls_from_topo(topo_dict:dict) -> dict:
    """
    Find block cylindrical coordinates of all the wall blocks

    Parameters
    ----------
    topo_dict: dict
        Dictionary input that describes the topology

    Returns
    ----------
    wall_dict: dict
        Dictionary of wall coordinates
    """ 

    r_wall = []
    l_wall = []
    try:
        elements = topo["Walls"]
        for element in elements:
            for block in elements[element]:
                r_wall.append(block["R"])
                l_wall.append(block["L"])
    except KeyError:
        pass

    wall_dict = {"r_wall": r_wall, "l_wall": l_wall}
    return wall_dict


def make_bound_from_topo(topo):
    BoundaryNames = []
    BoundaryType = []
    BoundaryRmin = []
    BoundaryRmax = []
    BoundaryLmin = []
    BoundaryLmax = []

    for boundary in topo["Boundary"]:
        BoundaryNames.append(boundary)
        tmp_bound_type = []
        tmp_rmin = []
        tmp_rmax = []
        tmp_lmin = []
        tmp_lmax = []
        for bound_element in topo["Boundary"][boundary]:
            tmp_bound_type.append(bound_element["type"])
            tmp_rmin.append(bound_element["Rmin"])
            tmp_rmax.append(bound_element["Rmax"])
            tmp_lmin.append(bound_element["Lmin"])
            tmp_lmax.append(bound_element["Lmax"])
        BoundaryType.append(tmp_bound_type)
        BoundaryRmin.append(tmp_rmin)
        BoundaryRmax.append(tmp_rmax)
        BoundaryLmin.append(tmp_lmin)
        BoundaryLmax.append(tmp_lmax)

    return {
        "names": BoundaryNames,
        "types": BoundaryType,
        "rmin": BoundaryRmin,
        "rmax": BoundaryRmax,
        "lmin": BoundaryLmin,
        "lmax": BoundaryLmax,
    }


def stretch_fun(G, N1):
    result = (1.0 - G) / (G * (1 - np.power(G, 1.0 / N1)))
    return result


# def stretch_fun(G,N1):
#    result = (1.0-G**(N1/(N1-1)))/((G**(1.0+1.0/(N1-1)))*(1-np.power(G,1.0/(N1-1))))
#    return result


def bissection(val, stretch_fun, N1):
    Gmin = 0.00001
    Gmax = 1000000
    resultmin = stretch_fun(Gmin, N1) - val
    resultmax = stretch_fun(Gmax, N1) - val
    if resultmin * resultmax > 0:
        error_msg = "Initial bounds of grading do not encompass the solution"
        logger.error(error_msg)
        raise ValueError(error_msg)

    for i in range(1000):
        Gmid = 0.5 * (Gmax + Gmin)
        resultmid = stretch_fun(Gmid, N1) - val
        if resultmid * resultmax < 0:
            Gmin = Gmid
            resultmin = resultmid
        else:
            Gmax = Gmid
            resultmax = resultmid

    return Gmid


def is_wall(l_wall:list[int], r_wall:list[int], ir:int, il:int)->int:
    """
    Is the present block a wall (not meshed)

    Parameters
    ----------
    l_wall: list[int]
        List of vertical coordinate index of wall blocks
    r_wall: list[int]
        List of radial coordinate index of wall blocks
    ir: int
        Radial coordinate index of the present block
    il: int
        Vertical coordinate index of the present block

    Returns
    ----------
    is_wall: int
        1 if block (ir,il) is a wall
        0 if block (ir, il) is fluid (and should be meshed)
    """
    is_wall = 0
    for iw in range(len(l_wall)):
        if l_wall[iw] == (il + 1) and r_wall[iw] == ir + 1:
            is_wall = 1
    return is_wall


def merge_and_sort(coord_list: list[float], reverse_coord: bool) -> list[float]:
    """
    Preprocess coordinates to obtain an ordered set of value ameable to block cylindrical meshing

    Parameters
    ----------
    coord_list: list[float]
        List of coordinates
    reverse_coord: bool
        Whether or not to reverse the coordinates 

    Returns
    ----------
    clean_list: list[float]
        preprocessed list

    """
    clean_list = list(set(coord_list))
    clean_list.sort(reverse=reverse_coord)
    return clean_list


def verticalCoarsening(
    ratio_properties, ref_block, NVert, L=None, gradVert=None, smooth=False
):
    if gradVert is None:
        gradVert = [1.0 for _ in range(len(NVert))]

    ratio_list = [entry["ratio"] for entry in ratio_properties]
    ratio_dir = []
    ratio_dir_ref = []
    for entry in ratio_properties:
        try:
            ratio_dir.append(entry["direction"])
        except KeyError:
            ratio_dir.append("+")
        try:
            ratio_dir_ref.append(entry["directionRef"])
        except KeyError:
            ratio_dir_ref.append("+")

    for iratio, ratio in enumerate(ratio_list):
        if abs(ratio - 1) < 1e-12:
            pass
        else:
            NVert[iratio] = max(int(round(NVert[iratio] * ratio)), 1)

    block_length = [abs(L[i] - L[i + 1]) for i in range(len(NVert))]
    block_cell_plus_length = [
        block_length[i] / NVert[i] for i in range(len(NVert))
    ]
    block_cell_minus_length = [
        block_length[i] / NVert[i] for i in range(len(NVert))
    ]

    if smooth:
        # Find which block to leave untouched
        indRef = int(ref_block)
        # Decide the order to follow
        indList = []
        for i in range(1, indRef + 1):
            indList.append(indRef - i)
        for i in range(indRef + 1, len(NVert)):
            indList.append(i)

        historyRatio = []
        for ind in indList:
            historyRatio.append(ratio_list[ind])
            if (
                abs(max(historyRatio) - 1) < 1e-12
                and abs(min(historyRatio) - 1) < 1e-12
            ):
                continue

            length = block_length[ind]

            if ratio_dir_ref[ind] == "-":
                deltaE = block_cell_plus_length[ind + 1]
            elif ratio_dir_ref[ind] == "+":
                deltaE = block_cell_minus_length[ind - 1]

            if ratio_dir[ind] == ratio_dir_ref[ind]:
                error_msg = (
                    f"Invalid coarsening ratio for vertical block {ind}"
                )
                error_msg += "\nratio dir and ratio dir ref must be opposite"
                logger.error(error_msg)
                raise ValueError(error_msg)

            if ratio_dir[ind] == "+":
                gradVert[ind] = 1.0 / bissection(
                    length / deltaE, stretch_fun, NVert[ind]
                )
                iterate = False
                origNVert = NVert[ind]
                while gradVert[ind] < 1 and NVert[ind] > 1:
                    iterate = True
                    NVert[ind] = max(
                        int(round(min(0.99 * NVert[ind], NVert[ind] - 1))), 1
                    )
                    gradVert[ind] = 1.0 / bissection(
                        length / deltaE, stretch_fun, NVert[ind]
                    )
                if iterate:
                    logger.warning(
                        f"reduced NVert[{ind}] from {origNVert} to {NVert[ind]}"
                    )
                block_cell_minus_length[ind] = deltaE
                block_cell_plus_length[ind] = deltaE * gradVert[ind]

            elif ratio_dir[ind] == "-":
                deltaE = block_cell_minus_length[ind - 1]
                gradVert[ind] = bissection(
                    length / deltaE, stretch_fun, NVert[ind]
                )

                iterate = False
                origNVert = NVert[ind]
                while gradVert[ind] > 1 and NVert[ind] > 1:
                    iterate = True
                    NVert[ind] = max(
                        int(round(min(0.99 * NVert[ind], NVert[ind] - 1))), 1
                    )
                    gradVert[ind] = bissection(
                        length / deltaE, stretch_fun, NVert[ind]
                    )
                if iterate:
                    logger.warning(
                        f"reduced NVert[{ind}] from {origNVert} to {NVert[ind]}"
                    )
                block_cell_minus_length[ind] = deltaE / gradVert[ind]
                block_cell_plus_length[ind] = deltaE

    minCell = np.amin(block_cell_minus_length)
    minCell = min(minCell, np.amin(block_cell_plus_length))
    maxCell = np.amax(block_cell_minus_length)
    maxCell = max(maxCell, np.amax(block_cell_plus_length))

    return NVert, gradVert, minCell, maxCell


def radialCoarsening(
    ratio_properties, ref_block, NR, R=None, gradR=None, smooth=False
):
    if gradR is None:
        gradR = [1.0 for _ in range(len(NR))]

    ratio_list = [entry["ratio"] for entry in ratio_properties]
    ratio_dir = []
    ratio_dir_ref = []
    for entry in ratio_properties:
        try:
            ratio_dir.append(entry["direction"])
        except KeyError:
            ratio_dir.append("+")
        try:
            ratio_dir_ref.append(entry["directionRef"])
        except KeyError:
            ratio_dir_ref.append("+")

    for iratio, ratio in enumerate(ratio_list):
        if abs(ratio - 1) < 1e-12:
            pass
        else:
            NR[iratio] = max(int(round(NR[iratio] * ratio)), 1)

    block_length = [R[0] / 2] + [
        abs(R[i] - R[i + 1]) for i in range(len(R) - 1)
    ]
    block_cell_plus_length = [block_length[i] / NR[i] for i in range(len(NR))]
    block_cell_minus_length = [block_length[i] / NR[i] for i in range(len(NR))]

    if smooth:
        # Find which block to leave untouched
        indRef = int(ref_block)
        # Decide the order to follow
        indList = []
        for i in range(1, indRef + 1):
            indList.append(indRef - i)
        for i in range(indRef + 1, len(NR)):
            indList.append(i)

        historyRatio = []
        for ind in indList:
            historyRatio.append(ratio_list[ind])
            if (
                abs(max(historyRatio) - 1) < 1e-12
                and abs(min(historyRatio) - 1) < 1e-12
            ):
                continue
            length = block_length[ind]

            if ratio_dir_ref[ind] == "-":
                deltaE = block_cell_plus_length[ind - 1]
            elif ratio_dir_ref[ind] == "+":
                deltaE = block_cell_minus_length[ind + 1]

            if ratio_dir[ind] == ratio_dir_ref[ind]:
                error_msg = f"Invalid coarsening ratio for radial block {ind}"
                error_msg += "\nratio dir and ratio dir ref must be opposite"
                logger.error(error_msg)
                raise ValueError(error_msg)

            if ratio_dir[ind] == "+":
                gradR[ind] = 1.0 / bissection(
                    length / deltaE, stretch_fun, NR[ind]
                )
                iterate = False
                origNR = NR[ind]
                while gradR[ind] < 1 and NR[ind] > 1:
                    iterate = True
                    NR[ind] = max(
                        int(round(min(0.99 * NR[ind], NR[ind] - 1))), 1
                    )
                    gradR[ind] = 1.0 / bissection(
                        length / deltaE, stretch_fun, NR[ind]
                    )
                if iterate:
                    logger.warning(
                        f"reduced NR[{ind}] from {origNR} to {NR[ind]}"
                    )
                block_cell_minus_length[ind] = deltaE
                block_cell_plus_length[ind] = deltaE * gradR[ind]

            elif ratio_dir[ind] == "-":
                deltaE = block_cell_minus_length[ind - 1]
                gradR[ind] = bissection(length / deltaE, stretch_fun, NR[ind])
                iterate = False
                origNR = NR[ind]
                while gradR[ind] > 1 and NR[ind] > 1:
                    iterate = True
                    NR[ind] = max(int(round(min(0.99 * NR[ind], -1))), 1)
                    gradR[ind] = bissection(
                        length / deltaE, stretch_fun, NR[ind]
                    )
                if iterate:
                    logger.warning(
                        f"reduced NR[{ind}] from {origNR} to {NR[ind]}"
                    )
                block_cell_minus_length[ind] = deltaE / gradR[ind]
                block_cell_plus_length[ind] = deltaE

    minCell = np.amin(block_cell_minus_length)
    minCell = min(minCell, np.amin(block_cell_plus_length))
    maxCell = np.amax(block_cell_minus_length)
    maxCell = min(maxCell, np.amax(block_cell_plus_length))

    # if smooth:
    #    if gradR is None or R is None:
    #        sys.exit(
    #            "ERROR: cannot smooth radial transition without grading list"
    #        )

    #    Length = R[last_R] - R[last_R - 1]
    #    deltaE = ((R[last_R - 1] - R[last_R - 2])) / NR[last_R - 1]
    #    gradR[last_R] = 1 / (
    #        bissection(Length / deltaE, stretch_fun, NR[last_R])
    #    )
    #    if (gradR[last_R] > 2 or gradR[last_R] < 0.5) and abs(
    #        ratio - 1
    #    ) <= 1e-12:
    #        logger.warning(
    #            "radial smoothing had to be used because your mesh is very coarse"
    #        )
    #        logger.warning("\tIncrease NS in input file to avoid this warning")

    return NR, gradR, minCell, maxCell
