import os
import pickle
import shutil

import numpy as np

from bird import BIRD_CASE_DIR
from bird.preprocess.json_gen.design_io import *
from bird.preprocess.json_gen.generate_designs import *

if __name__ == "__main__":
    # np.linspace(0.2,0.8, 4)?
    branchcom_spots = {}
    branchcom_spots[0] = np.linspace(0.2, 0.8, 4)
    branchcom_spots[1] = np.linspace(0.2, 0.8, 3)
    branchcom_spots[2] = np.linspace(0.2, 0.8, 4)

    # do sampling
    branches_com = [0, 1, 2]

    n_sim = 4

    config_dict = {}
    for i in range(n_sim):
        config_dict = sample(
            branches_com, branchcom_spots, config_dict=config_dict
        )

    vvm_l = [0.1, 1.6]
    pow_l = [1, 2]

    for vvm_v in vvm_l:
        vvm_str = str(vvm_v).replace(".", "_")
        for pow_v in pow_l:
            study_folder = f"study_{vvm_str}vvm_{pow_v}W"
            generate_small_reactor_cases(
                config_dict,
                branchcom_spots,
                vvm=vvm_v,
                power=pow_v,
                constantD=True,
                study_folder=study_folder,
            )
            write_script(f"{study_folder}/many_scripts", n_sim)
            write_prep(f"{study_folder}/prep.sh", n_sim)
            save_config_dict(f"{study_folder}/configs.pkl", config_dict)
            save_config_dict(
                f"{study_folder}/branchcom_spots.pkl", branchcom_spots
            )

    vvm_l = [0.1, 0.4]
    pow_l = [3000, 6000]

    for vvm_v in vvm_l:
        vvm_str = str(vvm_v).replace(".", "_")
        for pow_v in pow_l:
            study_folder = f"study_scaleup_{vvm_str}vvm_{pow_v}W"
            generate_scaledup_reactor_cases(
                config_dict,
                branchcom_spots,
                vvm=vvm_v,
                power=pow_v,
                constantD=True,
                study_folder=study_folder,
            )
            write_script(f"{study_folder}/many_scripts", n_sim)
            write_prep(f"{study_folder}/prep.sh", n_sim)
            save_config_dict(f"{study_folder}/configs.pkl", config_dict)
            save_config_dict(
                f"{study_folder}/branchcom_spots.pkl", branchcom_spots
            )
