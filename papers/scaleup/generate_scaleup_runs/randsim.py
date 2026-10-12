import os

import numpy as np

from bird.preprocess.json_gen.design_io import *
from bird.preprocess.json_gen.generate_designs import *


def optimization_setup():
    # spots on the branches where we can place sparger or mixers
    branchcom_spots = {}
    branchcom_spots[0] = np.linspace(0.2, 0.8, 4)
    branchcom_spots[1] = np.linspace(0.2, 0.8, 3)
    branchcom_spots[2] = np.linspace(0.2, 0.8, 4)
    # branches where the sparger and mixers are placed
    branches_com = [0, 1, 2]
    return branchcom_spots, branches_com


# Shared design pool
# once it is generated, all the other studies borrow this file
# This is useful to check if QOI values match and are correlated
DESIGN_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "designs.pkl"
)


if __name__ == "__main__":
    branchcom_spots, branches_com = optimization_setup()

    config_dict = load_or_sample_designs(
        DESIGN_FILE, branches_com, branchcom_spots, n_designs=400
    )

    # --- hyperparameters of the study
    vvm = 0.4
    constantD = True
    # Dynamic mixer properties
    # The sign business is because orientation depends on the branch
    # so that the mixer pushes the flow consistently
    mixer_params = {
        "Np": 6,
        "Vtip": 1.5,
        "sigma": 0.35,
        "radius": 0.4,
        "sign": {0: "+", 1: "+", 2: "-"},
        "swirl_sign": {0: "+", 1: "+", 2: "-"},
    }

    # --- Setup the scale up study
    # lev1 = 3.6 L, lev6 = 608 m3
    # 400 runs for lev 1 and lev 6
    # 200 runs for intermediate scales (match first 200 design variables
    # of the 400 pool)
    levels = [
        ("lev1", 0.05, 400),
        ("lev2", 0.1, 200),
        ("lev3", 0.2, 200),
        ("lev4", 0.4, 200),
        ("lev5", 0.8, 200),
        ("lev6", 2.7615275385627096, 400),
    ]

    # Per-level system/controlDict time-stepping.
    # Coarser scales (higher levels) tolerate initial dt
    level_controldict = {
        "lev1": {
            "maxCo": "0.25",
            "maxDeltaT": "0.00025",
            "deltaT": "1e-5",
            "endTime": "100",
        },
        "lev2": {
            "maxCo": "0.25",
            "maxDeltaT": "0.0005",
            "deltaT": "2e-5",
            "endTime": "200",
        },
        "lev3": {
            "maxCo": "0.25",
            "maxDeltaT": "0.001",
            "deltaT": "4e-5",
            "endTime": "400",
        },
        "lev4": {
            "maxCo": "0.25",
            "maxDeltaT": "0.002",
            "deltaT": "5e-5",
            "endTime": "400",
        },
        "lev5": {
            "maxCo": "0.25",
            "maxDeltaT": "0.004",
            "deltaT": "5e-5",
            "endTime": "400",
        },
        "lev6": {
            "maxCo": "0.25",
            "maxDeltaT": "0.005",
            "deltaT": "1e-4",
            "endTime": "400",
        },
    }

    # Per-level get_qoi.py parameters
    # gas density [kg/m3] at spargers change because of varying reactor height
    # c* (low, high) uniform-prior bounds on the CO2/H2 depend on scale
    level_qoi = {
        "lev1": {
            "rhog": 0.593333,
            "cstar_co2": (7.68, 8.22),
            "cstar_h2": (0.516, 0.548),
        },
        "lev2": {
            "rhog": 0.605962,
            "cstar_co2": (6.8, 7.93),
            "cstar_h2": (0.502, 0.514),
        },
        "lev3": {
            "rhog": 0.629974,
            "cstar_co2": (7.04, 8.06),
            "cstar_h2": (0.537, 0.557),
        },
        "lev4": {
            "rhog": 0.677935,
            "cstar_co2": (8.3, 9.8),
            "cstar_h2": (0.564, 0.601),
        },
        "lev5": {
            "rhog": 0.770533,
            "cstar_co2": (7.89, 10.2),
            "cstar_h2": (0.622, 0.653),
        },
        "lev6": {
            "rhog": 1.25,
            "cstar_co2": (14, 16.9),
            "cstar_h2": (1.04, 1.19),
        },
    }

    parent = "study"
    for name, scale, n_sim in levels:
        study_folder = os.path.join(parent, f"study_0_4vvm_{name}")
        generate_leveled_reactor_cases(
            config_dict,
            branchcom_spots,
            scale=scale,
            n_sim=n_sim,
            study_folder=study_folder,
            mixer_params=mixer_params,
            vvm=vvm,
            template_folder="./template",
            constantD=constantD,  # needs to be False if we do PBE
            start_time=1,
            account="gas2fuels",
            cores_per_sim=8,  # Pack as many sims per node as possible
            cores_per_node=104,
            walltime="120:00:00",
            controldict_params=level_controldict[name],
            rhog=level_qoi[name]["rhog"],
            cstar_co2=level_qoi[name]["cstar_co2"],
            cstar_h2=level_qoi[name]["cstar_h2"],
        )
