import json
import os
import pickle as pkl

import matplotlib as mpl
import numpy as np
from prettyPlot.plotting import *
from scipy.optimize import curve_fit


def get_sim_folds(path):
    folds = os.listdir(path)
    sim_folds = []
    for fold in folds:
        if fold.startswith("loop"):
            sim_folds.append(fold)
    return sim_folds


def func(t, cstar, kla):
    t = t
    t0 = 0
    c0 = 0
    return (cstar - c0) * (1 - np.exp(-kla * (t - t0))) + c0


def get_vl(verb=False):
    filename = os.path.join("constant", "globalVars")
    with open(filename, "r+") as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith("liqVol"):
            vol = float(line.split()[-1][:-1])
            break
    if verb:
        print(f"Read liqVol = {vol}m3")
    return vol


def get_vvm(verb=False):
    filename = os.path.join("constant", "globalVars")
    with open(filename, "r+") as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith("VVM"):
            vvm = float(line.split()[-1][:-1])
            break
    if verb:
        print(f"Read VVM = {vvm} [-]")
    return vvm


def get_As(verb=False):
    filename = os.path.join("constant", "globalVars")
    with open(filename, "r+") as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith("inletA"):
            As = float(line.split()[-1][:-1])
            break
    if verb:
        print(f"Read As = {As}m2")
    return As


def get_pmix(verb=False):
    with open("system/mixers.json", "r+") as f:
        data = json.load(f)
    mixer_list = data["mixers"]
    pmix = 0
    for mix in mixer_list:
        pmix += mix["power"] / 1000
    if verb:
        print(f"Read Mixing power = {pmix}kW")
    return pmix


def get_lh(verb=False):
    filename = os.path.join("system", "setFieldsDict")
    with open(filename, "r+") as f:
        lines = f.readlines()
    for line in lines:
        if "box (-1.0 -1.0 -1.0)" in line:
            height = float(line.split("(")[2].split()[1])
            break
    if verb:
        print(f"Read Height = {height}m")
    return height


def get_pinj(vvm, Vl, As, lh):
    rhog = 1.25  # kg /m3
    Vg = Vl * vvm / (60 * As * 1)  # m/s
    Ptank = 101325  # Pa
    # Ptank = 0 # Pa
    rhoL = 1000  # kg / m3
    Pl = 101325 + rhoL * 9.8 * lh  # Pa
    # W
    P1 = rhog * As * Vg**3
    # W
    P2 = (Pl - Ptank) * As * Vg
    # kg /s
    MF = rhog * Vg * As
    # kwh / kg
    e_m = (P1 + P2) / (3600 * 1000 * MF)

    # returns kW
    return (P1 + P2) * 1e-3


def get_qoi(kla_co2, cs_co2, kla_h2, cs_h2, verb=False):
    vvm = get_vvm(verb)
    As = get_As(verb)
    V_l = get_vl(verb)
    liqh = get_lh(verb)
    P_inj = get_pinj(vvm, V_l, As, liqh)
    P_mix = get_pmix(verb)

    qoi_kla_co2 = kla_co2 * cs_co2 * V_l * 0.04401
    qoi_kla_h2 = kla_h2 * cs_h2 * V_l * 0.002016 

    qoi_co2 = qoi_kla_co2 / (P_mix / 3600 + P_inj / 3600)
    qoi_h2 = qoi_kla_h2 / (P_mix / 3600 + P_inj / 3600)
    return qoi_co2 * qoi_h2, qoi_kla_co2*qoi_kla_h2


def get_qoi_uq(kla_co2, cs_co2, kla_h2, cs_h2):
    qoi = []
    qoi_kla = []
    for i in range(len(kla_co2)):
        if i == 0:
            verb = True
        else:
            verb = False
        qoi_tmp, qoi_kla_tmp = get_qoi(kla_co2[i], cs_co2[i], kla_h2[i], cs_h2[i], verb)
        qoi.append(qoi_tmp)
        qoi_kla.append(qoi_kla_tmp)
    qoi = np.array(qoi)
    qoi_kla = np.array(qoi_kla)
    return np.mean(qoi), np.std(qoi), np.mean(qoi_kla), np.std(qoi_kla)


os.makedirs("Figures", exist_ok=True)

dataFolder = "data"
fold = "local"

nuq = 100
#mean_cstar_co2 = np.random.uniform(12.6, 13.3, nuq)
#mean_cstar_h2 = np.random.uniform(0.902, 0.96, nuq)
mean_cstar_co2 = np.random.uniform(11.9, 13.4, nuq)
mean_cstar_h2 = np.random.uniform(0.884, 0.943, nuq)


tmp_cs_h2 = []
tmp_cs_co2 = []
tmp_kla_h2 = []
tmp_kla_co2 = []
cs_co2 = mean_cstar_co2
cs_h2 = mean_cstar_h2

a = np.load(os.path.join(dataFolder, fold, "conv.npz"))
endindex = -1
if (
    "c_h2" in a
    and "c_co2" in a
    and len(a["time"][:endindex] > 0)
    and (a["time"][:endindex][-1] > 95)
):
    for i in range(nuq):
        fitparamsH2, _ = curve_fit(
            func,
            np.array(a["time"][:endindex]),
            np.array(a["c_h2"][:endindex]),
            bounds=[(cs_h2[i] - 1e-6, 0), (cs_h2[i] + 1e-6, 1)],
        )
        fitparamsCO2, _ = curve_fit(
            func,
            np.array(a["time"][:endindex]),
            np.array(a["c_co2"][:endindex]),
            bounds=[(cs_co2[i] - 1e-6, 0), (cs_co2[i] + 1e-6, 1)],
        )
        tmp_kla_co2.append(fitparamsCO2[1])
        tmp_kla_h2.append(fitparamsH2[1])
        tmp_cs_h2.append(cs_h2[i])
        tmp_cs_co2.append(cs_co2[i])

qoi_m, qoi_s, qoi_kla_m, qoi_kla_s = get_qoi_uq(tmp_kla_co2, tmp_cs_co2, tmp_kla_h2, tmp_cs_h2)


with open("qoi.txt", "w+") as f:
    f.write(f"{qoi_m},{qoi_s}\n")

with open("qoi_kla.txt", "w+") as f:
    f.write(f"{qoi_kla_m},{qoi_kla_s}\n")
