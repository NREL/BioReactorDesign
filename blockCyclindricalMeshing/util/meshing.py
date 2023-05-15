import sys

import numpy as np


def make_walls_from_topo(topo):
    WallR = []
    WallL = []
    try:
        elements = topo["Walls"]
        for element in elements:
            for block in elements[element]:
                WallR.append(block["R"])
                WallL.append(block["L"])
    except KeyError:
        pass
    return {"WallR": WallR, "WallL": WallL}


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
    Gmin = 0.001
    Gmax = 1000
    resultmin = stretch_fun(Gmin, N1) - val
    resultmax = stretch_fun(Gmax, N1) - val
    if resultmin * resultmax > 0:
        print(
            "Error,the initial bounds of grading do not encompass the solution"
        )
        # stop

    for i in range(100):
        Gmid = 0.5 * (Gmax + Gmin)
        resultmid = stretch_fun(Gmid, N1) - val
        if resultmid * resultmax < 0:
            Gmin = Gmid
            resultmin = resultmid
        else:
            Gmax = Gmid
            resultmax = resultmid

    return Gmid


def amIwall(WallL, WallR, ir, il):
    # returns 1 if block is wall
    # returns 0 otherwise
    iwall = 0
    for iw in range(len(WallL)):
        if WallL[iw] == (il + 1) and WallR[iw] == ir + 1:
            iwall = 1
    return iwall


def mergeSort(list, reverse):
    list.sort(reverse=reverse)
    listtmp = []
    for val in list:
        if len(listtmp) == 0 or not val == listtmp[-1]:
            listtmp.append(val)
    return listtmp


def verticalOutletCoarsening(
    ratio, NVert, L=None, gradVert=None, smooth=False
):
    if ratio > 1:
        sys.exit("ERROR: vertical coarsening ratio should be < 1")

    if abs(ratio - 1) < 1e-12:
        return NVert, [1 for _ in range(len(NVert))]

    NVert[0] = int(NVert[0] * ratio)

    if smooth:
        if gradVert is None or L is None:
            sys.exit(
                "Error: cannot smooth vertical transition without grading list"
            )

        Length = L[0] - L[1]
        deltaE = (L[1] - L[2]) / NVert[1]
        gradVert[0] = 1 / (bissection(Length / deltaE, stretch_fun, NVert[0]))

        if (gradVert[0] > 2 or gradVert[0] < 0.5) and abs(ratio - 1) <= 1e-12:
            print(
                "WARNING: vertical smoothing had to be used because your mesh is very coarse"
            )
            print(
                "\tIncrease NVertSparger in input file to avoid this warning"
            )

    return NVert, gradVert


def radialFlowCoarseing(ratio, NR, R=None, gradR=None, smooth=False):
    if ratio > 1:
        sys.exit("ERROR: radial coarsening ratio should be < 1")
    if abs(ratio - 1) < 1e-12:
        return NR, [1 for _ in range(len(NR))]

    lastR = len(NR) - 1
    NR[lastR] = int(NR[lastR] * ratio)

    if smooth:
        if gradR is None or R is None:
            sys.exit(
                "ERROR: cannot smooth radial transition without grading list"
            )

        Length = R[lastR] - R[lastR - 1]
        deltaE = ((R[lastR - 1] - R[lastR - 2])) / NR[lastR - 1]
        gradR[lastR] = 1 / (
            bissection(Length / deltaE, stretch_fun, NR[lastR])
        )
        if (gradR[lastR] > 2 or gradR[lastR] < 0.5) and abs(
            ratio - 1
        ) <= 1e-12:
            print(
                "WARNING: radial smoothing had to be used because your mesh is very coarse"
            )
            print("\tIncrease NS in input file to avoid this warning")

    return NR, gradR
