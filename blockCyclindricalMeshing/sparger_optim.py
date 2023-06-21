import os
import sys

import numpy as np

sys.path.append("util")
from distutils.dir_util import copy_tree
from shutil import copy

import argument
from meshing import *
from modifyGeom import *
from myparser import parseJsonFile
from writeBlockMesh import *


def base_mesh(argsDict):
    geomDict = assemble_geom(argsDict)
    meshDict = assemble_mesh(argsDict, geomDict)
    writeBlockMeshDict(argsDict, geomDict, meshDict)


def generate_blockMeshDict(case_folder):
    argsDict = {
        "input_file": os.path.join(case_folder, "system", "input.json"),
        "topo_file": os.path.join(case_folder, "system", "topology.json"),
        "out_folder": os.path.join(case_folder, "system"),
    }
    base_mesh(argsDict)


def setupCaseFolder(target_folder, case_template_folder="case_template"):
    system_target_folder = os.path.join(target_folder, "system")
    os.makedirs(system_target_folder, exist_ok=True)
    copy_tree(
        os.path.join(case_template_folder, "system"), system_target_folder
    )
    constant_target_folder = os.path.join(target_folder, "constant")
    os.makedirs(constant_target_folder, exist_ok=True)
    copy_tree(
        os.path.join(case_template_folder, "constant"), constant_target_folder
    )
    orig_target_folder = os.path.join(target_folder, "0.orig")
    os.makedirs(orig_target_folder, exist_ok=True)
    copy_tree(os.path.join(case_template_folder, "0.orig"), orig_target_folder)

    copy(os.path.join(case_template_folder, "script.sh"), target_folder)
    copy(os.path.join(case_template_folder, "Allrun"), target_folder)


def side_sparger_variations(
    nCases,
    study_folder,
    case_template_folder="case",
    template_folder="template_sideSparger",
):
    os.makedirs(study_folder, exist_ok=True)
    heights = np.linspace(10, 200, nCases)
    np.savez(
        os.path.join(study_folder, "param_sideSparger.npz"), height=heights
    )
    for i in range(nCases):
        case_folder = os.path.join(study_folder, f"side_sparger_{i}")
        try:
            os.makedirs(case_folder)
        except OSError:
            sys.exit(f"ERROR: folder {case_folder} exists already")
        # Setup folder
        setupCaseFolder(case_folder, case_template_folder=case_template_folder)
        # Setup json files
        modify_sideSparger(
            heights[i], template_folder, os.path.join(case_folder, "system")
        )
        # Generate blockmesh
        generate_blockMeshDict(case_folder)

    gen_slurm_script(
        folder=study_folder, prefix="side_sparger_", nCases=nCases
    )
    gen_slurm_script_second(
        folder=study_folder, prefix="side_sparger_", nCases=nCases
    )


def flat_donut_variations(
    nCases,
    study_folder,
    case_template_folder="case",
    template_folder="template_flatDonut",
):
    os.makedirs(study_folder, exist_ok=True)
    widths = np.linspace(50, 200, nCases)
    np.savez(os.path.join(study_folder, "param_flatDonut.npz"), width=widths)
    for i in range(nCases):
        case_folder = os.path.join(study_folder, f"flat_donut_{i}")
        try:
            os.makedirs(case_folder)
        except OSError:
            sys.exit(f"ERROR: folder {case_folder} exists already")
        # Setup folder
        setupCaseFolder(case_folder, case_template_folder=case_template_folder)
        # Setup json files
        modify_flatDonut(
            widths[i], template_folder, os.path.join(case_folder, "system")
        )
        # Generate blockmesh
        generate_blockMeshDict(os.path.join(case_folder))

    gen_slurm_script(folder=study_folder, prefix="flat_donut_", nCases=nCases)
    gen_slurm_script_second(folder=study_folder, prefix="flat_donut_", nCases=nCases)


def multi_ring_variations(
    nCases,
    study_folder,
    case_template_folder="case",
    template_folder="template_multiRing",
):
    os.makedirs(study_folder, exist_ok=True)
    n_1D = round(np.sqrt(nCases))
    width = np.linspace(10, 50, n_1D)
    spacing = np.linspace(20, 100, n_1D)
    widthv, spacingv = np.meshgrid(width, spacing)
    nCases = n_1D * n_1D
    widths = np.ndarray.flatten(widthv)
    spacings = np.ndarray.flatten(spacingv)
    np.savez(
        os.path.join(study_folder, "param_multiRing.npz"),
        width=widths,
        spacing=spacings,
    )

    for i in range(nCases):
        case_folder = os.path.join(study_folder, f"multiRing_{i}")
        try:
            os.makedirs(case_folder)
        except OSError:
            sys.exit(f"ERROR: folder {case_folder} exists already")
        # Setup folder
        setupCaseFolder(case_folder, case_template_folder=case_template_folder)
        # Setup json files
        modify_multiring(
            widths[i],
            spacings[i],
            template_folder,
            os.path.join(case_folder, "system"),
        )
        # Generate blockmesh
        generate_blockMeshDict(os.path.join(case_folder))

    gen_slurm_script(folder=study_folder, prefix="multiRing_", nCases=nCases)
    gen_slurm_script_second(folder=study_folder, prefix="multiRing_", nCases=nCases)


def multi_ring_num_variations(
    study_folder,
    case_template_folder="case",
    template_root_folder=".",
):
    os.makedirs(study_folder, exist_ok=True)

    multiRing_template_folder = [
        "multiRing_simple2",
        "multiRing_simple3",
        "multiRing_simple4",
        "multiRing_simple5",
    ]
    multiRing_num = [2, 3, 4, 5]
    nCases = len(multiRing_template_folder)

    for i, (num, template_folder) in enumerate(
        zip(multiRing_num, multiRing_template_folder)
    ):
        case_folder = os.path.join(study_folder, f"multiRing_num_{i}")
        try:
            os.makedirs(case_folder)
        except OSError:
            sys.exit(f"ERROR: folder {case_folder} exists already")
        # Setup folder
        setupCaseFolder(case_folder, case_template_folder=case_template_folder)
        # Setup json files
        modify_multiring_num(
            os.path.join(template_root_folder, template_folder),
            os.path.join(case_folder, "system"),
        )
        # Generate blockmesh
        generate_blockMeshDict(os.path.join(case_folder))

    gen_slurm_script(
        folder=study_folder, prefix="multiRing_num_", nCases=nCases
    )
    gen_slurm_script_second(
        folder=study_folder, prefix="multiRing_num_", nCases=nCases
    )


def gen_slurm_script(folder, prefix, nCases):
    f = open(os.path.join(folder, "exec_first.sh"), "w+")
    for i in range(nCases):
        f.write(f"cd {prefix}{i}\n")
        f.write(f"sbatch script.sh\n")
        f.write(f"cd ..\n")
    f.close()
def gen_slurm_script_second(folder, prefix, nCases):
    f = open(os.path.join(folder, "exec_second.sh"), "w+")
    for i in range(nCases):
        f.write(f"cd {prefix}{i}\n")
        f.write(f"sbatch script_second.sh\n")
        f.write(f"cd ..\n")
    f.close()


if __name__ == "__main__":
    flat_donut_variations(
        10,
        "study_fine_flatDonut",
        case_template_folder="case_template",
        template_folder="template_flatDonut",
    )
    side_sparger_variations(
        10,
        "study_fine_sideSparger",
        case_template_folder="case_template",
        template_folder="template_sideSparger",
    )
    multi_ring_variations(
        10,
        "study_fine_multiRing",
        case_template_folder="case_template",
        template_folder="template_multiRing",
    )
    multi_ring_num_variations(
        "study_fine_multiRing_num",
        case_template_folder="case_template",
        template_root_folder=".",
    )
    flat_donut_variations(
        10,
        "study_coarse_flatDonut",
        case_template_folder="case_template",
        template_folder="template_flatDonut_coarse",
    )
    side_sparger_variations(
        10,
        "study_coarse_sideSparger",
        case_template_folder="case_template",
        template_folder="template_sideSparger_coarse",
    )
    multi_ring_variations(
        10,
        "study_coarse_multiRing",
        case_template_folder="case_template",
        template_folder="template_multiRing_coarse",
    )
