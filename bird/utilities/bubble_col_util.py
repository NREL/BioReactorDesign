import numpy as np

from bird.utilities.ofio import *


def readFromDict(val_dict, key, read_func=None, path=None, nCells=None):
    if key not in val_dict:
        field = read_func(path, nCells)
        val_dict[key] = field
    else:
        field = val_dict[key]
    return field, val_dict


def check_indLiq(ind_liq, cellCentres):
    height_liq = cellCentres[ind_liq, 1]
    ind_to_remove = np.argwhere(height_liq > 9.5)
    if len(ind_to_remove) > 0:
        ind_liq_copy = ind_liq.copy()
        n_remove = len(ind_to_remove)
        print(f"ind liq found to be at high heights {n_remove} times")
        ind_to_remove = list(ind_liq[ind_to_remove[:, 0]][:, 0])
        ind_liq_copy = list(set(list(ind_liq[:, 0])) - set(ind_to_remove))
        assert len(ind_liq_copy) == len(ind_liq) - n_remove
        ind_liq = np.reshape(np.array(ind_liq_copy), (-1, 1))
    return ind_liq


def check_indHeight(ind_height, cellCentres):
    height_liq = cellCentres[ind_height, 1]
    ind_to_remove = np.argwhere(height_liq < 6)
    if len(ind_to_remove) > 0:
        ind_height_copy = ind_height.copy()
        n_remove = len(ind_to_remove)
        print(f"ind height found to be at low heights {n_remove} times")
        ind_to_remove = list(ind_height[ind_to_remove[:, 0]][:, 0])
        ind_height_copy = list(
            set(list(ind_height_copy[:, 0])) - set(ind_to_remove)
        )
        assert len(ind_height_copy) == len(ind_height) - n_remove
        ind_height = np.reshape(np.array(ind_height_copy), (-1, 1))
    return ind_height


def indLiqFromDict(val_dict, localFolder, nCells, cellCentres):
    if "ind_liq" not in val_dict:
        alpha_gas, val_dict = readFromDict(
            val_dict=val_dict,
            key="alpha_gas",
            read_func=readOFScal,
            path=os.path.join(localFolder, "alpha.gas"),
            nCells=nCells,
        )
        ind_liq = np.argwhere(alpha_gas < 0.8)[:, 0]
        ind_liq = check_indLiq(ind_liq, cellCentres)
        val_dict["ind_liq"] = ind_liq
    else:
        ind_liq = val_dict["ind_liq"]

    return ind_liq, val_dict


def computeGH(localFolder, localFolder_vol, nCells, cellCentres, val_dict={}):
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
    ind_liq, val_dict = indLiqFromDict(
        val_dict, localFolder, nCells, cellCentres
    )

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

    tol = 1e-3
    tol_max = 0.1
    nFound = 0
    iteration = 0
    while nFound <= 10 and tol < tol_max:
        ind_height = np.argwhere(abs(alpha_gas - 0.8) < tol)
        ind_height = check_indHeight(ind_height, cellCentres)
        nFound = len(ind_height)
        tol *= 1.1
        tol = np.clip(tol, a_min=None, a_max=0.2)
        iteration += 1

    if iteration > 1:
        print(f"\tChanged GH tol to {tol:.2g}")

    height_liq = np.mean(cellCentres[ind_height, 1])
    holdup = height_liq / height_liq_base - 1

    return holdup, val_dict


def computeDiam(localFolder, nCells, cellCentres, val_dict={}):
    d_gas, val_dict = readFromDict(
        val_dict=val_dict,
        key="d_gas",
        read_func=readOFScal,
        path=os.path.join(localFolder, "d.gas"),
        nCells=nCells,
    )
    ind_liq, val_dict = indLiqFromDict(
        val_dict, localFolder, nCells, cellCentres
    )

    diam = np.mean(d_gas[ind_liq])

    return diam, val_dict


def computeSpec_liq(
    localFolder, nCells, field_name, key, cellCentres, val_dict={}
):
    species, val_dict = readFromDict(
        val_dict=val_dict,
        key=key,
        read_func=readOFScal,
        path=os.path.join(localFolder, field_name),
        nCells=nCells,
    )
    ind_liq, val_dict = indLiqFromDict(
        val_dict, localFolder, nCells, cellCentres
    )

    species = np.mean(species[ind_liq])

    return species, val_dict


def computeSpec_kla_field(
    localFolder, nCells, key_suffix, cellCentres, val_dict={}, diff=None
):
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
        if diff is None:
            T_gas, val_dict = readFromDict(
                val_dict=val_dict,
                key="T_gas",
                read_func=readOFScal,
                path=os.path.join(localFolder, "T.gas"),
                nCells=nCells,
            )
            mu = 1.67212e-06 * np.sqrt(T_gas) / (1 + 170.672 / T_gas)
            D = mu / rho_gas / 0.7
        else:
            D = np.ones(rho_gas.shape) * diff
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

    return kla, val_dict


def computeSpec_kla(
    localFolder, nCells, key_suffix, cellCentres, val_dict={}, diff=None
):
    kla, val_dict = computeSpec_kla_field(
        localFolder, nCells, key_suffix, cellCentres, val_dict, diff
    )

    ind_liq, val_dict = indLiqFromDict(
        val_dict, localFolder, nCells, cellCentres
    )

    return np.mean(kla[ind_liq]), val_dict
