import os
import pickle
import shutil

import numpy as np
from bird.preprocess.json_gen.design_io import *

from bird import BIRD_CASE_DIR


def compare_config(config1, config2):
    same = True
    for key in config1:
        if np.linalg.norm(config1[key] - config2[key]) > 1e-6:
            same = False
            return same
    return same


def check_config(config):
    success = False
    inlet_exist = False
    for key in config:
        if len(np.argwhere(config[key] == 1)) > 0:
            inlet_exist = True
            break
    if inlet_exist:
        success = True
    else:
        success = False
    return success


def save_config_dict(filename, config_dict):
    with open(filename, "wb") as f:
        pickle.dump(config_dict, f)


def load_config_dict(filename):
    with open(filename, "rb") as f:
        config_dict = pickle.load(f)
    return config_dict


def write_script_start(filename, n):
    with open(filename, "w+") as f:
        for i in range(n):
            f.write(f"cd Sim_{i}\n")
            f.write(f"sbatch script\n")
            f.write(f"cd ..\n")

def write_script_post(filename, n):
    with open(filename, "w+") as f:
        for i in range(n):
            f.write(f"cd Sim_{i}\n")
            f.write(f"sbatch script_post\n")
            f.write(f"cd ..\n")


def write_prep(filename, n):
    with open(filename, "w+") as f:
        f.write("prep () {\n")
        f.write(f"\tcd $1\n")
        f.write(f"\treconstructPar -newTimes\n")
        f.write(f"\tcd ..\n")
        f.write("}\n")
        f.write(f"\n")
        f.write(
            f"source /projects/gas2fuels/ofoam_cray_mpich/OpenFOAM-dev/etc/bashrc\n"
        )
        for i in range(n):
            f.write(f"prep Sim_{i}\n")


def overwrite_vvm(case_folder, vvm):
    list_dir = os.listdir(case_folder)
    if not "constant" in list_dir:
        sys.exit(
            f"ERROR: {case_folder} is likely not a case folder, could not find constant/"
        )
    else:
        filename = os.path.join(case_folder, "constant", "globalVars_temp")
        filename_write = os.path.join(
            case_folder, "constant", "globalVars_temp2"
        )
        with open(filename, "r+") as f:
            lines = f.readlines()
        with open(filename_write, "w+") as f:
            for line in lines:
                if line.startswith("VVM"):
                    f.write(f"VVM\t{vvm};\n")
                else:
                    f.write(line)
        shutil.copy(
            os.path.join(case_folder, "constant", "globalVars_temp2"),
            os.path.join(case_folder, "constant", "globalVars_temp"),
        )
        os.remove(os.path.join(case_folder, "constant", "globalVars_temp2"))


def overwrite_bubble_size_model(case_folder, constantD=False):
    list_dir = os.listdir(case_folder)
    if not "constant" in list_dir:
        sys.exit(
            f"ERROR: {case_folder} is likely not a case folder, could not find constant/"
        )
    else:
        filename = os.path.join(case_folder, "presteps.sh")
        filename_write = os.path.join(case_folder, "presteps2.sh")
        with open(filename, "r+") as f:
            lines = f.readlines()
        with open(filename_write, "w+") as f:
            for line in lines:
                if line.startswith("cp constant/phaseProperties"):
                    if constantD:
                        f.write(
                            "cp constant/phaseProperties_constantd constant/phaseProperties\n"
                        )
                    else:
                        f.write(
                            "cp constant/phaseProperties_pbe constant/phaseProperties\n"
                        )
                else:
                    f.write(line)
        shutil.copy(
            os.path.join(case_folder, "presteps2.sh"),
            os.path.join(case_folder, "presteps.sh"),
        )
        os.remove(os.path.join(case_folder, "presteps2.sh"))


def generate_small_reactor_cases(
    config_dict, branchcom_spots, vvm, power, constantD, study_folder
):
    templateFolder = "loop_reactor_pbe_dynmix_nonstat_headbranch"

    geom_dict = make_default_geom_dict_from_file(
        f"{BIRD_CASE_DIR}/{templateFolder}/system/mesh.json",
        rescale=0.05,
    )
    try:
        shutil.rmtree(study_folder)
    except:
        pass
    os.makedirs(study_folder)
    for simid in config_dict:
        shutil.copytree(
            f"{BIRD_CASE_DIR}/{templateFolder}",
            f"{study_folder}/Sim_{simid}",
        )
        bc_dict = {}
        bc_dict["inlets"] = []
        bc_dict["outlets"] = []
        bc_dict["outlets"].append(
            {
                "branch_id": 6,
                "type": "circle",
                "frac_space": 1,
                "normal_dir": 1,
                "radius": 0.4,
                "nelements": 50,
                "block_pos": "top",
            }
        )
        bc_dict["outlets"].append(
            {
                "branch_id": 4,
                "type": "circle",
                "frac_space": 1,
                "normal_dir": 1,
                "radius": 0.4,
                "nelements": 50,
                "block_pos": "top",
            }
        )
        for branch in config_dict:
            if branch in [0, 1, 2]:
                ind = np.argwhere(config_dict[simid][branch] == 1)
                if len(ind) > 0:
                    ind = list(ind[:, 0])
                    for iind in ind:
                        bc_dict["inlets"].append(
                            {
                                "branch_id": branch,
                                "type": "circle",
                                "frac_space": branchcom_spots[branch][iind],
                                "normal_dir": 1,
                                "radius": 0.4,
                                "nelements": 50,
                                "block_pos": "bottom",
                            }
                        )
        generate_stl_patch(
            os.path.join(
                study_folder, f"Sim_{simid}", "system", "inlets_outlets.json"
            ),
            bc_dict,
            geom_dict,
        )

        mix_list = []
        for branch in config_dict:
            if branch in [0, 1, 2]:
                ind = np.argwhere(config_dict[simid][branch] == 0)
                if len(ind) > 0:
                    ind = list(ind[:, 0])
                    for iind in ind:
                        if branch == 0:
                            sign = "+"
                        else:
                            sign = "-"
                        mix_list.append(
                            {
                                "branch_id": branch,
                                "frac_space": branchcom_spots[branch][iind],
                                "start_time": 1,
                                "power": power,
                                "sign": sign,
                            }
                        )
        generate_dynamic_mixer(
            os.path.join(
                study_folder, f"Sim_{simid}", "system", "mixers.json"
            ),
            mix_list,
            geom_dict,
        )
        overwrite_vvm(
            case_folder=os.path.join(study_folder, f"Sim_{simid}"), vvm=vvm
        )
        overwrite_bubble_size_model(
            case_folder=os.path.join(study_folder, f"Sim_{simid}"),
            constantD=constantD,
        )

    geom_dict = make_default_geom_dict_from_file(
        f"{BIRD_CASE_DIR}/{templateFolder}/system/mesh.json",
        rescale=0.05,
    )


def generate_scaledup_reactor_cases(
    config_dict, branchcom_spots, vvm, power, constantD, study_folder
):

    templateFolder = "loop_reactor_pbe_dynmix_nonstat_headbranch_scaleup"
    geom_dict = make_default_geom_dict_from_file(
        f"{BIRD_CASE_DIR}/{templateFolder}/system/mesh.json"
    )
    try:
        shutil.rmtree(study_folder)
    except:
        pass
    os.makedirs(study_folder)
    for simid in config_dict:
        shutil.copytree(
            f"{BIRD_CASE_DIR}/{templateFolder}",
            f"{study_folder}/Sim_{simid}",
        )
        bc_dict = {}
        bc_dict["inlets"] = []
        bc_dict["outlets"] = []
        bc_dict["outlets"].append(
            {
                "branch_id": 6,
                "type": "circle",
                "frac_space": 1,
                "normal_dir": 1,
                "radius": 0.4,
                "nelements": 50,
                "block_pos": "top",
            }
        )
        bc_dict["outlets"].append(
            {
                "branch_id": 4,
                "type": "circle",
                "frac_space": 1,
                "normal_dir": 1,
                "radius": 0.4,
                "nelements": 50,
                "block_pos": "top",
            }
        )
        for branch in config_dict:
            if branch in [0, 1, 2]:
                ind = np.argwhere(config_dict[simid][branch] == 1)
                if len(ind) > 0:
                    ind = list(ind[:, 0])
                    for iind in ind:
                        bc_dict["inlets"].append(
                            {
                                "branch_id": branch,
                                "type": "circle",
                                "frac_space": branchcom_spots[branch][iind],
                                "normal_dir": 1,
                                "radius": 0.4,
                                "nelements": 50,
                                "block_pos": "bottom",
                            }
                        )
        generate_stl_patch(
            os.path.join(
                study_folder, f"Sim_{simid}", "system", "inlets_outlets.json"
            ),
            bc_dict,
            geom_dict,
        )

        mix_list = []
        for branch in config_dict:
            if branch in [0, 1, 2]:
                ind = np.argwhere(config_dict[simid][branch] == 0)
                if len(ind) > 0:
                    ind = list(ind[:, 0])
                    for iind in ind:
                        if branch == 0:
                            sign = "+"
                        else:
                            sign = "-"
                        mix_list.append(
                            {
                                "branch_id": branch,
                                "frac_space": branchcom_spots[branch][iind],
                                "start_time": 3,
                                "power": power,
                                "sign": sign,
                            }
                        )
        generate_dynamic_mixer(
            os.path.join(
                study_folder, f"Sim_{simid}", "system", "mixers.json"
            ),
            mix_list,
            geom_dict,
        )
        overwrite_vvm(
            case_folder=os.path.join(study_folder, f"Sim_{simid}"), vvm=vvm
        )
        overwrite_bubble_size_model(
            case_folder=os.path.join(study_folder, f"Sim_{simid}"),
            constantD=constantD,
        )
