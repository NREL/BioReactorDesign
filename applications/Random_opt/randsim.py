import os
import pickle
import shutil

import numpy as np

from bird import BIRD_CASE_DIR
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

def random_sample(branches_com, branchcom_spots, config_dict={}):
    config = {}
    # choices = ["mix", "sparger", "none"]
    choices_com = [0, 1, 2]
    for branch in branches_com:
        config[branch] = np.random.choice(
            choices_com, size=len(branchcom_spots[branch])
        )

    existing = False
    new_config_key = 0
    for old_key_conf in config_dict:
        if compare_config(config_dict[old_key_conf], config):
            existing = True
            print("FOUND SAME CONFIG")
            return config_dict
        new_config_key = old_key_conf + 1

    if check_config(config):
        config_dict[new_config_key] = config

    return config_dict

if __name__ == "__main__":
    branchcom_spots, branches_com = optimization_setup()
    n_sim = 4
    config_dict = {}
    for i in range(n_sim):
        config_dict = random_sample(
            branches_com, branchcom_spots, config_dict=config_dict
        )

    vvm_l = [0.4]
    pow_l = [2]

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
            write_script_start(f"{study_folder}/many_scripts_start", n_sim)
            write_script_post(f"{study_folder}/many_scripts_post", n_sim)
            write_prep(f"{study_folder}/prep.sh", n_sim)
            save_config_dict(f"{study_folder}/configs.pkl", config_dict)
            save_config_dict(
                f"{study_folder}/branchcom_spots.pkl", branchcom_spots
            )

    vvm_l = [0.1, 0.4]
    pow_l = [6000]

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
            write_script_start(f"{study_folder}/many_scripts", n_sim)
            write_script_post(f"{study_folder}/many_scripts", n_sim)
            write_prep(f"{study_folder}/prep.sh", n_sim)
            save_config_dict(f"{study_folder}/configs.pkl", config_dict)
            save_config_dict(
                f"{study_folder}/branchcom_spots.pkl", branchcom_spots
            )
