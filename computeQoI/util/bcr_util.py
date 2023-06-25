import numpy as np
from ofio import *


def readFromDict(val_dict, key, read_func=None, path=None, nCells=None):
    if key not in val_dict:
        field = read_func(path, nCells)
        val_dict[key] = field
    else:
        field = val_dict[key]
    return field, val_dict


def indLiqFromDict(val_dict, localFolder, nCells):
    if "ind_liq" not in val_dict:
        alpha_gas, val_dict = readFromDict(
            val_dict=val_dict,
            key="alpha_gas",
            read_func=readOFScal,
            path=os.path.join(localFolder, "alpha.gas"),
            nCells=nCells,
        )
        ind_liq = np.argwhere(alpha_gas < 0.9)[:, 0]
        val_dict["ind_liq"] = ind_liq
    else:
        ind_liq = val_dict["ind_liq"]

    return ind_liq, val_dict


def computeGH(localFolder, localFolder_vol, nCells, val_dict={}):
    alpha_gas, val_dict = readFromDict(
        val_dict=val_dict,
        key="alpha_gas",
        read_func=readOFScal,
        path=os.path.join(localFolder, "alpha.gas"),
        nCells=nCells,
    )
    volume, val_dict = readFromDict(
        val_dict=val_dict,
        key="volume",
        read_func=readOFScal,
        path=os.path.join(localFolder_vol, "V"),
        nCells=nCells,
    )
    ind_liq, val_dict = indLiqFromDict(val_dict, localFolder, nCells)

    holdup = np.sum(alpha_gas[ind_liq] * volume[ind_liq]) / np.sum(
        volume[ind_liq]
    )
    return holdup, val_dict


def computeGH_height(
    localFolder, nCells, cellCentres, height_liq_base, val_dict={}
):
    alpha_gas, val_dict = readFromDict(
        val_dict=val_dict,
        key="alpha_gas",
        read_func=readOFScal,
        path=os.path.join(localFolder, "alpha.gas"),
        nCells=nCells,
    )

    ind_height = np.argwhere(abs(alpha_gas - 0.9) < 1e-2)
    height_liq = np.mean(cellCentres[ind_height, 1])
    holdup = height_liq / height_liq_base - 1

    return holdup, val_dict


def computeDiam(localFolder, nCells, val_dict={}):
    d_gas, val_dict = readFromDict(
        val_dict=val_dict,
        key="d_gas",
        read_func=readOFScal,
        path=os.path.join(localFolder, "d.gas"),
        nCells=nCells,
    )
    ind_liq, val_dict = indLiqFromDict(val_dict, localFolder, nCells)

    diam = np.mean(d_gas[ind_liq])

    return diam, val_dict


def computeSpec_liq(localFolder, nCells, field_name, key, val_dict={}):
    species, val_dict = readFromDict(
        val_dict=val_dict,
        key=key,
        read_func=readOFScal,
        path=os.path.join(localFolder, field_name),
        nCells=nCells,
    )
    ind_liq, val_dict = indLiqFromDict(val_dict, localFolder, nCells)

    species = np.mean(species[ind_liq])

    return species, val_dict


def computeSpec_kla(localFolder, nCells, key_suffix, val_dict={}):
    if "slip_vel" not in val_dict:
        u_gas, val_dict = readFromDict(
            val_dict=val_dict,
            key="u_gas",
            read_func=readOFVec,
            path=os.path.join(localFolder, "U.gas"),
            nCells=nCells,
        )
        u_liq, val_dict = readFromDict(
            val_dict=val_dict,
            key="u_liq",
            read_func=readOFVec,
            path=os.path.join(localFolder, "U.liquid"),
            nCells=nCells,
        )
        slipvel = np.linalg.norm(u_liq - u_gas, axis=1)
        val_dict["slipvel"] = slipvel

    rho_gas, val_dict = readFromDict(
        val_dict=val_dict,
        key="rho_gas",
        read_func=readOFScal,
        path=os.path.join(localFolder, "thermo:rho.gas"),
        nCells=nCells,
    )
    if "D_" + key_suffix not in val_dict:
        T_gas, val_dict = readFromDict(
            val_dict=val_dict,
            key="T_gas",
            read_func=readOFScal,
            path=os.path.join(localFolder, "T.gas"),
            nCells=nCells,
        )
        mu = 1.67212e-06 * np.sqrt(T_gas) / (1 + 170.672 / T_gas)
        D = mu / rho_gas / 0.7
        val_dict["D_" + key_suffix] = D
    else:
        D = val_dict["D_" + key_suffix]

    d_gas, val_dict = readFromDict(
        val_dict=val_dict,
        key="d_gas",
        read_func=readOFScal,
        path=os.path.join(localFolder, "d.gas"),
        nCells=nCells,
    )
    if "Sh_" + key_suffix not in val_dict:
        # Sh = 1.12*np.sqrt(rho_gas*slipvel*d_gas/(D*0.7*rho_gas))*np.sqrt(0.7)
        Sh = (
            2.0
            + 0.552
            * np.sqrt(rho_gas * slipvel * d_gas / (D * 0.7 * rho_gas))
            * 0.8889
        )
        val_dict["Sh_" + key_suffix] = Sh
    else:
        Sh = val_dict["Sh_" + key_suffix]

    alpha_gas, val_dict = readFromDict(
        val_dict=val_dict,
        key="alpha_gas",
        read_func=readOFScal,
        path=os.path.join(localFolder, "alpha.gas"),
        nCells=nCells,
    )

    kla = Sh * 6 * alpha_gas / d_gas / d_gas * D

    ind_liq, val_dict = indLiqFromDict(val_dict, localFolder, nCells)

    return np.mean(kla[ind_liq]), val_dict
