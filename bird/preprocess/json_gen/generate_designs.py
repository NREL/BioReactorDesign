from design_io import *
import numpy as np
import pickle
import os

def compare_config(config1, config2):
    same = True
    for key in config1:
        if np.linalg.norm(config1[key] - config2[key])>1e-6:
            same = False
            return same
    return same

def check_config(config):
    success = False
    inlet_exist = False
    for key in config:
        if len(np.argwhere(config[key]==1))>0:
            inlet_exist = True
            break
    if inlet_exist:
        success = True
    else:
        success = False
    return success

def save_config_dict(filename, config_dict):
    with open(filename, "wb") as f:
        pickle.dump( config_dict, f)

def load_config_dict(filename):
    with open(filename, "rb") as f:
        config_dict = pickle.load(f)
    return config_dict

def sample(branches_com, branches_mix, branchcom_spots, branchmix_spots, config_dict={}):
    config = {}
    #choices = ["mix", "sparger", "none"]
    choices_com = [0, 1, 2]
    choices_mix = [0, 2]
    for branch in branches_com:
        config[branch] = np.random.choice(choices_com, size=len(branchcom_spots[branch]))
    for branch in branches_mix:
        config[branch] = np.random.choice(choices_mix, size=len(branchmix_spots[branch]))
 
    existing = False
    new_config_key = 0 
    for old_key_conf in config_dict:
        if compare_config(config_dict[old_key_conf], config):
            existing = True
            return None
        new_config_key = old_key_conf+1

    if check_config(config):
        config_dict[new_config_key] = config
    return config_dict


def generate_designs(config_dict, branchcom_spots, branchmix_spots):
    geom_dict = make_default_geom_dict_from_file('/Users/mhassana/Desktop/GitHub/BioReactorDesign_Aug8/OFsolvers/tutorial_cases/loop_reactor_pbe_dynmix_nonstat_headbranch_scaleup/system/mesh.json')
    for simid in config_dict:
        os.makedirs(str(simid), exist_ok=True)
        bc_dict = {}
        bc_dict['inlets'] = []
        bc_dict['outlets'] = []
        bc_dict['outlets'].append({"branch_id": 6, "type": "circle", "frac_space": 1, "normal_dir": 1, "radius": 0.4, "nelements": 50, "block_pos": "top"})
        bc_dict['outlets'].append({"branch_id": 4, "type": "circle", "frac_space": 1, "normal_dir": 1, "radius": 0.4, "nelements": 50, "block_pos": "top"})
        for branch in config_dict:
            if branch in [0, 1, 2]:
                ind = np.argwhere(config_dict[simid][branch]==1)
                if len(ind)>0:
                    ind=list(ind[:,0])
                    for iind in ind:
                        bc_dict['inlets'].append({"branch_id": branch, "type": "circle", "frac_space": branchcom_spots[branch][iind], "normal_dir": 1, "radius": 0.4, "nelements": 50, "block_pos": "bottom"})
        generate_stl_patch(os.path.join(str(simid), 'inlets_outlets.json'), bc_dict, geom_dict)

        mix_list = []
        for branch in config_dict:
            if branch in [0, 1, 2]:
                ind = np.argwhere(config_dict[simid][branch]==0)
                if len(ind)>0:
                    ind=list(ind[:,0])
                    for iind in ind:
                        mix_list.append({"branch_id": branch, "frac_space": branchcom_spots[branch][iind], "start_time": 4, "power": 3000, "sign": "+"})
            if branch in [3,7]:
                ind = np.argwhere(config_dict[simid][branch]==0)
                if len(ind)>0:
                    ind=list(ind[:,0])
                    for iind in ind:
                        mix_list.append({"branch_id": branch, "frac_space": branchmix_spots[branch][iind], "start_time": 4, "power": 3000, "sign": "+"})
        generate_dynamic_mixer(os.path.join(str(simid), 'mixers.json'), mix_list, geom_dict)


branchcom_spots = {}
branchcom_spots[0] = np.linspace(0.08,0.92, 8)  
branchcom_spots[1] = np.linspace(0.08,0.92, 4)
branchcom_spots[2] = np.linspace(0.08,0.92, 8)

branchmix_spots = {}
branchmix_spots[7] = np.linspace(0.2,0.8, 3)
branchmix_spots[3] = np.linspace(0.2,0.8, 3)


# do sampling
branches_com = [0, 1, 2]
branches_mix = [3, 7]

config_dict = {}
for i in range(20):
    config_dict = sample(branches_com, branches_mix, branchcom_spots, branchmix_spots, config_dict=config_dict)
save_config_dict('configs.pkl', config_dict)
save_config_dict("branchcom_spots.pkl", branchcom_spots)
save_config_dict("branchmix_spots.pkl", branchmix_spots)
generate_designs(config_dict, branchcom_spots, branchmix_spots)
