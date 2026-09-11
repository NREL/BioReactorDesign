import json
import os
import pickle
import shutil
from pathlib import Path

import numpy as np

from bird import BIRD_CASE_DIR, BIRD_DIR, logger
from bird.meshing.block_rect_mesh import from_block_rect_to_seg
from bird.preprocess.dynamic_mixer.mixer import (
    ActuatorMixer,
    actuator_disk_power,
)
from bird.preprocess.json_gen.design_io import *


def id2simfolder(sim_id: int) -> str:
    """
    Generates simulation folder name from simulation index

    Parameters
    ----------
    sim_id: int
        Simulation index

    Returns
    ----------
    sim_folder : str
        Simulation folder name
    """
    sim_folder = f"Sim_{sim_id:04}"
    return sim_folder


def compare_config(config1, config2):
    same = True
    for key in config1:
        if np.linalg.norm(config1[key] - config2[key]) > 1e-6:
            same = False
            return same
    return same


def check_config(config):
    """Accept a design only if it has at least one sparger.

    Choice value ``1`` marks a sparger (a bottom gas inlet); ``0`` a mixer,
    ``2`` nothing. A design with no sparger is rejected, so
    :func:`sample_placement_designs` never keeps one.
    """
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


def sample_placement_designs(
    branches_com,
    branchcom_spots,
    n_designs,
    choices=(0, 1, 2),
    max_attempts=1_000_000,
):
    """Randomly sample `n_designs` distinct, valid placement designs.

    Draws are non-deterministic (the caller must NOT seed for reproducibility).
    Each design maps ``branch_id -> array`` of per-spot choices;
    :func:`check_config` keeps only designs with at least one inlet and
    :func:`compare_config` rejects duplicates. Keys are contiguous ``0..n-1``.

    Parameters
    ----------
    branches_com:
        Branch ids on which choices are placed.
    branchcom_spots:
        ``branch_id -> array`` of candidate spot fractions.
    n_designs:
        Number of distinct valid designs to return.
    choices:
        Per-spot categorical choices (e.g. mixer/sparger/none).
    max_attempts:
        Give up after this many draws.
    """
    config_dict = {}
    attempts = 0
    while len(config_dict) < n_designs and attempts < max_attempts:
        config = {
            b: np.random.choice(choices, size=len(branchcom_spots[b]))
            for b in branches_com
        }
        attempts += 1
        if any(compare_config(config_dict[k], config) for k in config_dict):
            continue
        if check_config(config):
            config_dict[len(config_dict)] = config
    if len(config_dict) < n_designs:
        raise RuntimeError(
            f"only found {len(config_dict)} designs in {attempts} attempts"
        )
    return config_dict


def load_or_sample_designs(
    design_file,
    branches_com,
    branchcom_spots,
    n_designs,
    choices=(0, 1, 2),
):
    """Borrow designs from `design_file` if it exists, else sample and save.

    The first sweep run samples a fresh random design set (see
    :func:`sample_placement_designs`) and pickles it to `design_file`; every
    later sweep pointed at the same file loads it instead of re-sampling, so
    ``Sim_i`` is the same design across all sweeps without relying on a fixed
    seed. `design_file` should hold enough designs for the largest sweep --
    downstream slicing (``sorted(config_dict)[:n_sim]``) selects the first
    `n_sim`.
    """
    if os.path.exists(design_file):
        logger.info(f"Borrowing designs from existing {design_file}")
        return load_config_dict(design_file)
    logger.info(f"Sampling {n_designs} designs and saving to {design_file}")
    config_dict = sample_placement_designs(
        branches_com, branchcom_spots, n_designs, choices=choices
    )
    save_config_dict(design_file, config_dict)
    return config_dict


def write_script_start(filename, n):
    with open(filename, "w+") as f:
        for i in range(n):
            sim_folder = id2simfolder(i)
            f.write(f"cd {sim_folder}\n")
            f.write(f"sbatch script\n")
            f.write(f"cd ..\n")


def write_script_post(filename, n):
    with open(filename, "w+") as f:
        for i in range(n):
            sim_folder = id2simfolder(i)
            f.write(f"cd {sim_folder}\n")
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
            sim_folder = id2simfolder(i)
            f.write(f"prep {sim_folder}\n")


def overwrite_vvm(case_folder, vvm):
    list_dir = os.listdir(case_folder)
    if not "constant" in list_dir:
        error_msg = f"{case_folder} is likely not a case folder, could not find constant/"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
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
        error_msg = f"{case_folder} is likely not a case folder, could not find constant/"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
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
    config_dict,
    branchcom_spots,
    vvm,
    power,
    constantD,
    study_folder,
    template_folder="loop_reactor_pbe_dynmix_nonstat_headbranch",
):
    if not os.path.isabs(template_folder):

        template_folder = os.path.join(
            f"{BIRD_CASE_DIR}", f"{template_folder}"
        )

    geom_dict = make_default_geom_dict_from_file(
        os.path.join(f"{template_folder}", "system", "mesh.json"),
        rescale=0.05,
    )
    try:
        shutil.rmtree(study_folder)
    except:
        pass
    Path(study_folder).mkdir(parents=True, exist_ok=True)
    for sim_id in config_dict:
        sim_folder = id2simfolder(sim_id)
        shutil.copytree(
            f"{template_folder}",
            os.path.join(f"{study_folder}", sim_folder),
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
                ind = np.argwhere(config_dict[sim_id][branch] == 1)
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
                study_folder, sim_folder, "system", "inlets_outlets.json"
            ),
            bc_dict,
            geom_dict,
        )

        mix_list = []
        for branch in config_dict:
            if branch in [0, 1, 2]:
                ind = np.argwhere(config_dict[sim_id][branch] == 0)
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
            os.path.join(study_folder, sim_folder, "system", "mixers.json"),
            mix_list,
            geom_dict,
        )
        overwrite_vvm(
            case_folder=os.path.join(study_folder, sim_folder), vvm=vvm
        )
        overwrite_bubble_size_model(
            case_folder=os.path.join(study_folder, sim_folder),
            constantD=constantD,
        )

    geom_dict = make_default_geom_dict_from_file(
        os.path.join(f"{template_folder}", "system", "mesh.json"),
        rescale=0.05,
    )


def generate_scaledup_reactor_cases(
    config_dict,
    branchcom_spots,
    vvm,
    power,
    constantD,
    study_folder,
    template_folder="loop_reactor_pbe_dynmix_nonstat_headbranch_scaleup",
):

    if not os.path.isabs(template_folder):
        template_folder = os.path.join(
            f"{BIRD_CASE_DIR}", f"{template_folder}"
        )

    geom_dict = make_default_geom_dict_from_file(
        os.path.join(f"{template_folder}", "system", "mesh.json")
    )
    try:
        shutil.rmtree(study_folder)
    except:
        pass
    Path(study_folder).mkdir(parents=True, exist_ok=True)
    for sim_id in config_dict:
        sim_folder = id2simfolder(sim_id)
        shutil.copytree(
            f"{template_folder}",
            os.path.join(f"{study_folder}", sim_folder),
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
                ind = np.argwhere(config_dict[sim_id][branch] == 1)
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
                study_folder, sim_folder, "system", "inlets_outlets.json"
            ),
            bc_dict,
            geom_dict,
        )

        mix_list = []
        for branch in config_dict:
            if branch in [0, 1, 2]:
                ind = np.argwhere(config_dict[sim_id][branch] == 0)
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
            os.path.join(study_folder, sim_folder, "system", "mixers.json"),
            mix_list,
            geom_dict,
        )
        overwrite_vvm(
            case_folder=os.path.join(study_folder, sim_folder), vvm=vvm
        )
        overwrite_bubble_size_model(
            case_folder=os.path.join(study_folder, sim_folder),
            constantD=constantD,
        )


def check_sparger_config(
    sparger_locs: list[float],
    n_spargers: int | None,
    sparger_spacing: float,
    edge_spacing: float,
    n_branches: int,
    bypass_sparger_spacing: bool,
) -> None:
    """
    Check realizability of the sparger placement configuration

    Parameters
    ----------
    sparger_locs : list[float]
        Location of every sparger along the loop reactor coordinate [-]
        There are 3 branches. Spargers can be placed anywhere
        between edge_spacing and (1-edge_spacing) fractions of the branch
        Each sparger locations must be between 0 and 3*1=3
    n_spargers : int|None
        Number of spargers
    sparger_spacing : float
        Spacing between two spargers [-]
    edge_spacing : float
        Spacing required between any sparger and the edges of the branches [-]
    n_branches : int
        Number of loop reactor branches
    bypass_sparger_spacing: bool
        If true, allow an overlap of spargers
    """

    # Check that number of spargers is consistent
    if n_spargers is None:
        n_spargers = len(sparger_locs)
    else:
        assert n_spargers == len(sparger_locs)
    assert n_spargers >= 1

    # Basis check on the number of branches
    assert n_branches > 0

    # Check that locations of spargers is consistent
    # There are n_branches branches. Spargers can be placed anywhere
    # between edge_spacing and (1-edge_spacing) fractions of the branch
    # Each sparger locations must be between 0 and n_branches*1=n_branches
    assert edge_spacing > 0
    assert edge_spacing < 1
    assert all(np.array(sparger_locs) >= 0)
    assert all(np.array(sparger_locs) <= float(n_branches))
    for ibranch in range(n_branches):
        if ibranch == 0:
            assert not np.any(np.array(sparger_locs) < edge_spacing)
        if ibranch == n_branches - 1:
            assert not np.any(
                np.array(sparger_locs) > float(n_branches) - edge_spacing
            )
        assert not np.any(
            (np.array(sparger_locs) > float(ibranch) + 1.0 - edge_spacing)
            & (np.array(sparger_locs) < float(ibranch) + 1.0 + edge_spacing)
        )

    # Check that spargers are sufficiently spaced out
    assert sparger_spacing >= 0
    if not bypass_sparger_spacing:
        assert all(np.diff(np.sort(np.array(sparger_locs))) >= sparger_spacing)


def generate_single_scaledup_reactor_sparger_cases(
    sparger_locs: list[float],
    n_spargers: int | None = None,
    sparger_spacing: float = 0.15,
    edge_spacing: float = 0.2,
    n_branches: int = 3,
    sim_id: int = 0,
    constantD: bool = True,
    vvm: float = 0.4,
    study_folder: str = ".",
    template_folder: str = "loop_reactor_pbe_dynmix_nonstat_headbranch_scaleup",
    bypass_sparger_spacing: bool = False,
):
    """
    Generates loop reactor case with desired sparger placement configuration

    Parameters
    ----------
    sparger_locs : list[float]
        Location of every sparger along the loop reactor coordinate [-]
    n_spargers : int|None
        Number of spargers
    sparger_spacing : float
        Spacing between two spargers [-]
    edge_spacing : float
        Spacing required between any sparger and the edges of the branches [-]
    n_branches : int
        Number of loop reactor branches
    sim_id : int
        Index identifier of the simulation
    constantD : bool
        If true, use constant bubble diameter
        If false, use population balance
    vvm : float
        VVM value [-]
    study_folder : str
        Where to generate the case
    template_folder: str
        The case template to start from
    bypass_sparger_spacing: bool
        If true, allow an overlap of spargers
    """

    # Sanity checks
    check_sparger_config(
        sparger_locs=sparger_locs,
        n_spargers=n_spargers,
        sparger_spacing=sparger_spacing,
        edge_spacing=edge_spacing,
        n_branches=n_branches,
        bypass_sparger_spacing=bypass_sparger_spacing,
    )

    # Find on which branch is each sparger
    branch_id = [int(entry) for entry in sparger_locs]

    # Case generation
    if not os.path.isabs(template_folder):

        template_folder = os.path.join(
            f"{BIRD_CASE_DIR}", f"{template_folder}"
        )
    geom_dict = make_default_geom_dict_from_file(
        os.path.join(f"{template_folder}", "system", "mesh.json")
    )

    # Start from template
    sim_folder = id2simfolder(sim_id)
    shutil.copytree(
        f"{template_folder}",
        os.path.join(f"{study_folder}", sim_folder),
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

    for branch, loc in zip(branch_id, sparger_locs):
        bc_dict["inlets"].append(
            {
                "branch_id": branch,
                "type": "circle",
                "frac_space": loc - branch,
                "normal_dir": 1,
                "radius": 0.4,
                "nelements": 50,
                "block_pos": "bottom",
            }
        )

    generate_stl_patch(
        os.path.join(
            study_folder, sim_folder, "system", "inlets_outlets.json"
        ),
        bc_dict,
        geom_dict,
    )

    mix_list = []
    generate_dynamic_mixer(
        os.path.join(study_folder, sim_folder, "system", "mixers.json"),
        mix_list,
        geom_dict,
    )
    overwrite_vvm(case_folder=os.path.join(study_folder, sim_folder), vvm=vvm)
    overwrite_bubble_size_model(
        case_folder=os.path.join(study_folder, sim_folder),
        constantD=constantD,
    )


def overwrite_scale(case_folder, scale):
    """Rewrite the ``transformPoints`` scale in presteps.sh to `scale`."""
    filename = os.path.join(case_folder, "presteps.sh")
    with open(filename, "r+") as f:
        lines = f.readlines()
    with open(filename, "w+") as f:
        for line in lines:
            if line.strip().startswith("transformPoints"):
                f.write(f'transformPoints "scale=({scale} {scale} {scale})"\n')
            else:
                f.write(line)


# setFields liquid-init box upper corner in UNSCALED (per-block) units: y is the
# 4-block liquid fill height; x/z are made wide to cover the whole domain.
_SETFIELDS_BOX_UPPER = (200.0, 4.0, 200.0)


def overwrite_setfields_box(case_folder, scale):
    """Rewrite the setFields liquid-init box to `scale` * ``_SETFIELDS_BOX_UPPER``.

    setFields runs after transformPoints, so the box lives in scaled
    coordinates; the lower corner stays ``(-1 -1 -1)``.
    """
    bx, by, bz = (v * scale for v in _SETFIELDS_BOX_UPPER)
    filename = os.path.join(case_folder, "system", "setFieldsDict")
    with open(filename, "r+") as f:
        lines = f.readlines()
    with open(filename, "w+") as f:
        for line in lines:
            if line.strip().startswith("box "):
                indent = line[: len(line) - len(line.lstrip())]
                f.write(f"{indent}box (-1.0 -1.0 -1.0) ({bx} {by} {bz});\n")
            else:
                f.write(line)


def overwrite_qoi_params(case_folder, rhog, cstar_co2, cstar_h2):
    """Rewrite the per-level QoI parameters in get_qoi.py.

    These are hardcoded per dimension in get_qoi.py, so each level needs its
    own values (the mixer power is NOT touched here -- get_qoi.py reads it from
    mixers.json).

    Parameters
    ----------
    rhog:
        Gas density in :math:`kg.m^{-3}` used in the injection-power estimate.
    cstar_co2:
        (low, high) uniform-prior bounds for the CO2 c*.
    cstar_h2:
        (low, high) uniform-prior bounds for the H2 c*.
    """
    co2_lo, co2_hi = cstar_co2
    h2_lo, h2_hi = cstar_h2
    filename = os.path.join(case_folder, "get_qoi.py")
    with open(filename, "r+") as f:
        lines = f.readlines()
    with open(filename, "w+") as f:
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("rhog ="):
                indent = line[: len(line) - len(line.lstrip())]
                f.write(f"{indent}rhog = {rhog}  # kg /m3\n")
            elif stripped.startswith("mean_cstar_co2 ="):
                f.write(
                    "mean_cstar_co2 = "
                    f"np.random.uniform({co2_lo}, {co2_hi}, nuq)\n"
                )
            elif stripped.startswith("mean_cstar_h2 ="):
                f.write(
                    "mean_cstar_h2 = "
                    f"np.random.uniform({h2_lo}, {h2_hi}, nuq)\n"
                )
            else:
                f.write(line)


def overwrite_ncores(case_folder, n):
    """Rewrite ``numberOfSubdomains`` in system/decomposeParDict to `n`."""
    filename = os.path.join(case_folder, "system", "decomposeParDict")
    with open(filename, "r+") as f:
        lines = f.readlines()
    with open(filename, "w+") as f:
        for line in lines:
            if line.strip().startswith("numberOfSubdomains"):
                f.write(f"numberOfSubdomains {n};\n")
            else:
                f.write(line)


def overwrite_controldict(case_folder, params):
    """Rewrite time-stepping entries in system/controlDict.

    Parameters
    ----------
    params:
        Dict with any of ``deltaT``, ``endTime``, ``maxCo``, ``maxDeltaT``;
        each present key overwrites its scalar entry.
    """
    filename = os.path.join(case_folder, "system", "controlDict")
    with open(filename, "r+") as f:
        lines = f.readlines()
    with open(filename, "w+") as f:
        for line in lines:
            key = line.strip().split(None, 1)[0] if line.strip() else ""
            if key in params:
                f.write(f"{key:<16}{params[key]};\n")
            else:
                f.write(line)


def write_script_single(
    case_folder,
    account="gas2fuels",
    cores=4,
    solver="birdmultiphaseEulerFoam",
    walltime="47:59:00",
):
    """Write a per-case SLURM script (``script_single``) running one case."""
    ofbashrc = "/projects/gas2fuels/ofoam_cray_mpich/OpenFOAM-dev/etc/bashrc"
    with open(os.path.join(case_folder, "script_single"), "w+") as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --job-name=lev_single\n")
        f.write("#SBATCH --nodes=1\n")
        f.write(f"#SBATCH --ntasks-per-node={cores}\n")
        f.write(f"#SBATCH --time={walltime}\n")
        f.write(f"#SBATCH --account={account}\n\n")
        f.write("bash presteps.sh\n")
        f.write(f"source {ofbashrc}\n")
        f.write("decomposePar -fileHandler collated\n")
        f.write(f"srun -n {cores} {solver} -parallel -fileHandler collated\n")
        f.write("reconstructPar -newTimes\n")


def write_script_post_single(case_folder, account="gas2fuels"):
    """Write a per-case post-processing SLURM script (``script_post_single``)."""
    with open(os.path.join(case_folder, "script_post_single"), "w+") as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --job-name=lev_post\n")
        f.write("#SBATCH --nodes=1\n")
        f.write("#SBATCH --ntasks-per-node=1\n")
        f.write("#SBATCH --time=00:59:00\n")
        f.write(f"#SBATCH --account={account}\n\n")
        f.write("bash computeQOI.sh\n")


def write_foam_stub(case_folder: str) -> None:
    """Create an empty ``test.foam`` so ParaView can open the case."""
    open(os.path.join(case_folder, "test.foam"), "w").close()


def write_pack_scripts(
    study_folder,
    sim_ids,
    sims_per_node=26,
    cores_per_sim=4,
    account="gas2fuels",
    solver="birdmultiphaseEulerFoam",
    walltime="47:59:00",
):
    """Write node-packing scripts (Option A): pack_XXX bundles + submit_all.sh.

    Each bundle runs up to `sims_per_node` cases concurrently on one node, each
    via ``srun --exclusive -n cores_per_sim`` (so `sims_per_node*cores_per_sim`
    cores are used per node).
    """
    ofbashrc = "/projects/gas2fuels/ofoam_cray_mpich/OpenFOAM-dev/etc/bashrc"
    bundles = [
        sim_ids[i : i + sims_per_node]
        for i in range(0, len(sim_ids), sims_per_node)
    ]
    pack_names = []
    for b, bundle in enumerate(bundles):
        pack_name = f"pack_{b:03}"
        pack_names.append(pack_name)
        with open(os.path.join(study_folder, pack_name), "w+") as f:
            f.write("#!/bin/bash\n")
            f.write(f"#SBATCH --job-name=lev_{pack_name}\n")
            f.write("#SBATCH --nodes=1\n")
            f.write("#SBATCH --exclusive\n")
            f.write(f"#SBATCH --time={walltime}\n")
            f.write(f"#SBATCH --account={account}\n\n")
            f.write(f"source {ofbashrc}\n\n")
            f.write("run_sim () {\n")
            f.write('\tcd "$1"\n')
            f.write("\tbash presteps.sh > log.presteps 2>&1\n")
            f.write(
                "\tdecomposePar -fileHandler collated > log.decompose 2>&1\n"
            )
            f.write(
                f"\tsrun --exclusive -n {cores_per_sim} {solver} -parallel"
                " -fileHandler collated > log.solve 2>&1\n"
            )
            f.write("\treconstructPar -newTimes > log.reconstruct 2>&1\n")
            f.write("\tcd ..\n")
            f.write("}\n\n")
            for sim_id in bundle:
                f.write(f"run_sim {id2simfolder(sim_id)} &\n")
            f.write("wait\n")
    with open(os.path.join(study_folder, "submit_all.sh"), "w+") as f:
        for pack_name in pack_names:
            f.write(f"sbatch {pack_name}\n")


def write_pack_post_scripts(
    study_folder,
    sim_ids,
    sims_per_node=26,
    account="gas2fuels",
    walltime="1:00:00",
):
    """Write post-processing packing scripts: pack_post_XXX + submit_all_post.sh.

    Mirrors :func:`write_pack_scripts` one-to-one (same `sims_per_node`
    bundling, so ``pack_post_b`` post-processes exactly the sims in
    ``pack_b``), but each sim runs the QoI pipeline on a single core:
    ``reconstructPar`` then ``read_history.py`` + ``get_qoi.py`` under the
    bird_mixer conda env. The bundle requests one core per sim
    (``ntasks-per-node = len(bundle)``) and its sims run concurrently, each in
    its own subshell so their conda state stays isolated. ``submit_all_post.sh``
    sbatches every bundle.
    """
    ofbashrc = "/projects/gas2fuels/ofoam_cray_mpich/OpenFOAM-dev/etc/bashrc"
    conda_env = "/projects/gas2fuels/conda_env/bird_mixer/"
    bundles = [
        sim_ids[i : i + sims_per_node]
        for i in range(0, len(sim_ids), sims_per_node)
    ]
    pack_names = []
    for b, bundle in enumerate(bundles):
        pack_name = f"pack_post_{b:03}"
        pack_names.append(pack_name)
        with open(os.path.join(study_folder, pack_name), "w+") as f:
            f.write("#!/bin/bash\n")
            f.write(f"#SBATCH --job-name=lev_{pack_name}\n")
            f.write("#SBATCH --nodes=1\n")
            f.write(f"#SBATCH --ntasks-per-node={len(bundle)}\n")
            f.write(f"#SBATCH --time={walltime}\n")
            f.write(f"#SBATCH --account={account}\n\n")
            f.write("run_post () {\n")
            f.write("(\n")
            f.write('\tcd "$1"\n')
            f.write(f"\tsource {ofbashrc}\n")
            f.write("\treconstructPar -newTimes > log.reconstruct 2>&1\n")
            f.write("\tmodule load conda\n")
            f.write(f"\tconda activate {conda_env}\n")
            f.write(
                "\tpython read_history.py -cr .. -cn local -df data"
                " > log.readhist 2>&1\n"
            )
            f.write("\tpython get_qoi.py > log.getqoi 2>&1\n")
            f.write("\tconda deactivate\n")
            f.write(") &\n")
            f.write("}\n\n")
            for sim_id in bundle:
                f.write(f"run_post {id2simfolder(sim_id)}\n")
            f.write("wait\n")
    with open(os.path.join(study_folder, "submit_all_post.sh"), "w+") as f:
        for pack_name in pack_names:
            f.write(f"sbatch {pack_name}\n")


def generate_leveled_reactor_cases(
    config_dict,
    branchcom_spots,
    scale,
    n_sim,
    study_folder,
    mixer_params,
    vvm=0.4,
    constantD=True,
    start_time=3,
    rhog=None,
    cstar_co2=None,
    cstar_h2=None,
    template_folder="loop_reactor_pbe_dynmix_nonstat_headbranch_scaleup",
    account="gas2fuels",
    cores_per_sim=16,
    cores_per_node=128,
    controldict_params=None,
    walltime="47:59:00",
):
    """Generate one scale level of the actuator-disk (ball) design sweep.

    One template drives every level; the level `scale` is applied both to the
    mixers.json rescale (mixer positions) and to presteps.sh transformPoints.
    Uses the first `n_sim` designs of `config_dict`, so ``Sim_i`` is the same
    design at every level. `mixer_params` holds Np/Vtip/sigma/radius and the
    per-branch `sign` and `swirl_sign` (each a dict keyed by branch_id), which
    are written verbatim into each mixer entry of mixers.json.

    `rhog`, `cstar_co2` and `cstar_h2` are the per-level QoI parameters; when
    given they are written into each case's get_qoi.py (see
    :func:`overwrite_qoi_params`). Left as ``None`` the template get_qoi.py is
    used unchanged.

    Each sim runs on `cores_per_sim` cores; the node-packing bundles fit
    ``cores_per_node // cores_per_sim`` sims per node. `walltime` is the SLURM
    ``--time`` written into both the per-case and node-packing scripts.

    `controldict_params`, when given, is a dict of ``system/controlDict``
    scalar entries (any of ``deltaT``, ``endTime``, ``maxCo``, ``maxDeltaT``)
    written into every case of this level via :func:`overwrite_controldict`;
    left ``None`` the template controlDict is used unchanged.
    """
    # Resolve template_folder: use it if it points at a real directory (an
    # absolute path or one relative to the cwd, e.g. "./template_kom");
    # otherwise fall back to a template bundled under data_case_gen.
    bundled = os.path.join(
        BIRD_DIR, "preprocess", "data_case_gen", template_folder
    )
    if os.path.isdir(template_folder):
        template_folder = os.path.abspath(template_folder)
    elif os.path.isdir(bundled):
        template_folder = bundled
    else:
        raise FileNotFoundError(
            f"template_folder not found: {template_folder!r} is not a "
            f"directory, nor is {bundled!r}"
        )
    geom_dict = make_default_geom_dict_from_file(
        os.path.join(template_folder, "system", "mesh.json")
    )
    # mesh.json has no rescale; force the level scale (matched by transformPoints)
    for a in ("x", "y", "z"):
        geom_dict["Geometry"]["OverallDomain"][a]["rescale"] = scale
    seg_geom = from_block_rect_to_seg(geom_dict["Geometry"])
    model = {
        "volumetric_source": "ball",
        "power": "from_Np_Vtip",
        "momentum_source": "axial_and_swirl",
    }

    try:
        shutil.rmtree(study_folder)
    except FileNotFoundError:
        pass
    Path(study_folder).mkdir(parents=True, exist_ok=True)

    sim_ids = sorted(config_dict)[:n_sim]
    for sim_id in sim_ids:
        sim_folder = id2simfolder(sim_id)
        case = os.path.join(study_folder, sim_folder)
        shutil.copytree(template_folder, case)

        # Outlets are template-driven: read from the template's
        # inlets_outlets.json so each head shape carries its own outlet -- the
        # square-head templates define a full-face rectangle covering the whole
        # top patch, while the baseline templates define the two disk outlets on
        # branches 6 and 4. Inlets stay config-driven, built below.
        with open(
            os.path.join(template_folder, "system", "inlets_outlets.json")
        ) as f:
            bc_dict = {
                "inlets": [],
                "outlets": json.load(f).get("outlets", []),
            }
        for branch in (0, 1, 2):
            for iind in np.argwhere(config_dict[sim_id][branch] == 1)[:, 0]:
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
            os.path.join(case, "system", "inlets_outlets.json"),
            bc_dict,
            geom_dict,
        )

        mix_list = []
        for branch in (0, 1, 2):
            for iind in np.argwhere(config_dict[sim_id][branch] == 0)[:, 0]:
                # sign / swirl_sign are per-branch (keyed by branch_id) so the
                # axial push and swirl form one coherent loop circulation; the
                # values live in mixer_params and are written to mixers.json.
                frac = branchcom_spots[branch][iind]
                # derive this mixer's power from Np/Vtip and its (scaled) radius
                probe = ActuatorMixer()
                probe.update_from_loop_dict(
                    {
                        "branch_id": branch,
                        "frac_space": frac,
                        "radius": mixer_params["radius"],
                    },
                    seg_geom,
                )
                power = actuator_disk_power(
                    mixer_params["Np"], mixer_params["Vtip"], probe.R
                )
                mix_list.append(
                    {
                        "branch_id": branch,
                        "frac_space": float(frac),
                        "start_time": start_time,
                        "sign": mixer_params["sign"][branch],
                        "swirl_sign": mixer_params["swirl_sign"][branch],
                        "radius": mixer_params["radius"],
                        "Vtip": mixer_params["Vtip"],
                        "Np": mixer_params["Np"],
                        "sigma": mixer_params["sigma"],
                        "power": power,
                    }
                )
        generate_dynamic_mixer(
            os.path.join(case, "system", "mixers.json"),
            mix_list,
            geom_dict,
            model=model,
        )
        overwrite_vvm(case_folder=case, vvm=vvm)
        overwrite_scale(case_folder=case, scale=scale)
        overwrite_setfields_box(case_folder=case, scale=scale)
        if rhog is not None and cstar_co2 is not None and cstar_h2 is not None:
            overwrite_qoi_params(
                case_folder=case,
                rhog=rhog,
                cstar_co2=cstar_co2,
                cstar_h2=cstar_h2,
            )
        overwrite_ncores(case_folder=case, n=cores_per_sim)
        if controldict_params is not None:
            overwrite_controldict(case_folder=case, params=controldict_params)
        overwrite_bubble_size_model(case_folder=case, constantD=constantD)
        write_script_single(
            case, account=account, cores=cores_per_sim, walltime=walltime
        )
        write_script_post_single(case, account=account)
        write_foam_stub(case)

    # pack as many sims per node as the requested cores allow
    sims_per_node = max(1, cores_per_node // cores_per_sim)
    write_pack_scripts(
        study_folder,
        sim_ids,
        sims_per_node=sims_per_node,
        cores_per_sim=cores_per_sim,
        account=account,
        walltime=walltime,
    )
    # post-processing packs mirror the run packs one-to-one (1 core per sim)
    write_pack_post_scripts(
        study_folder,
        sim_ids,
        sims_per_node=cores_per_node,
        account=account,
    )
    write_prep(os.path.join(study_folder, "prep.sh"), n_sim)
    save_config_dict(os.path.join(study_folder, "configs.pkl"), config_dict)
    save_config_dict(
        os.path.join(study_folder, "branchcom_spots.pkl"), branchcom_spots
    )
