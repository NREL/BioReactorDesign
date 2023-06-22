import numpy as np
from ofio import *


def readFromDict(val_dict, key, read_func=None, path=None, nCells=None):
    if key not in val_dict:
        field = read_func(path, nCells)
        val_dict[key] = field
    else:
        field = val_dict[key]
    return field, val_dict


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

    if "Sh_" + key_suffix not in val_dict:
        d_gas, val_dict = readFromDict(
            val_dict=val_dict,
            key="d_gas",
            read_func=readOFScal,
            path=os.path.join(localFolder, "d.gas"),
            nCells=nCells,
        )
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
