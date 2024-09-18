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


def check_ga_samples(ga_samples):
    # we expect ga_samples of dimension (N, 11), where N is the number of samples in the batch
    assert len(ga_samples.shape) == 2
    assert ga_samples.shape[1] == 11
    assert np.amax(ga_samples) <= 2
    assert np.amin(ga_samples) >= 0


def ga2sim(ga_samples):
    branchcom_spots, branches_com = optimization_setup()
    check_ga_samples(ga_samples)
    config_dict = {}
    # Split ga_samples across the 3 branches
    split_ga_samples = np.split(ga_samples, [4, 7], axis=1)
    n_batch = ga_samples.shape[0]
    for i_batch in range(n_batch):
        config = {}
        for branch in branches_com:
            config[branch] = split_ga_samples[branch][i_batch, :]
        config_dict[i_batch] = config

    return config_dict

def setup_sim_batch(ga_samples, vvm_v=0.4, pow_v=6000, study_folder='GA_batch'):
    # GA_batch
    check_ga_samples(ga_samples)
    n_batch = ga_samples.shape[0]
    branchcom_spots, branches_com = optimization_setup()

    # Generate cases
    config_dict = ga2sim(ga_samples)
    generate_scaledup_reactor_cases(
        config_dict,
        branchcom_spots=branchcom_spots,
        vvm=vvm_v,
        power=pow_v,
        constantD=True,
        study_folder=study_folder,
    )
    write_script_start(f"{study_folder}/many_scripts_start", n_batch)
    write_script_post(f"{study_folder}/many_scripts_post", n_batch)
    write_prep(f"{study_folder}/prep.sh", n_batch)
    save_config_dict(f"{study_folder}/configs.pkl", config_dict)
    save_config_dict(f"{study_folder}/branchcom_spots.pkl", branchcom_spots)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GA setup")
    parser.add_argument(
        "-bf",
        "--batch_folder",
        type=str,
        metavar="",
        required=False,
        help="Folder that contains the batch of sims",
        default="GAbatch_0.4vvm_6kW"
    )
    parser.add_argument(
        "-vvm",
        "--vvm",
        type=float,
        metavar="",
        required=False,
        help="Volume flow rate",
        default=0.4
    )    
    parser.add_argument(
        "-pow",
        "--power",
        type=float,
        metavar="", 
        required=False,
        help="mixer power",
        default=6000
    )
    parser.add_argument(
        "-gas",
        "--ga_sample_file",
        type=str,
        metavar="", 
        required=False,
        help="file containing the ga samples to execute",
        default=None
    )   
    args, unknown = parser.parse_known_args()     
   
    if args.ga_sample_file is None:
        n_batch = 3
        ga_samples = np.random.choice([0, 1, 2], size=(n_batch, 11))
    else:
        ga_samples = np.load(args.ga_sample_file)
    vvm_v = args.vvm
    pow_v = args.power
    study_folder = args.batch_folder
    setup_sim_batch(ga_samples, vvm_v=vvm_v, pow_v=pow_v, study_folder=study_folder)
