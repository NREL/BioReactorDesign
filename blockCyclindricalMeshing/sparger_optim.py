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


def setupCaseFolder(target_folder, case_template_folder="case"):
    system_target_folder = os.path.join(target_folder, "system")
    os.makedirs(system_target_folder, exist_ok=True)
    copy_tree(
        os.path.join(case_template_folder, "system"), system_target_folder
    )
    copy(os.path.join(case_template_folder, "test.foam"), target_folder)


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


def flat_donut_variations(
    nCases,
    study_folder,
    case_template_folder="case",
    template_folder="template_flatDonut",
):
    os.makedirs(study_folder, exist_ok=True)
    widths = np.linspace(30, 200, nCases)
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


if __name__ == "__main__":
    # side_sparger_variations(10, 'study', template_folder='template_sideSparger')
    flat_donut_variations(10, "study", template_folder="template_flatDonut")
    # multi_ring_variations(10, 'study', template_folder='template_multiRing')
