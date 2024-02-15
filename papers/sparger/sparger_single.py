import os
import sys
from distutils.dir_util import copy_tree
from shutil import copy, rmtree

import numpy as np
from modifyGeom import *

from bird.meshing.block_cyl_mesh import *


def base_mesh(input_file, topo_file, output_folder):
    geomDict = assemble_geom(input_file, topo_file)
    meshDict = assemble_mesh(input_file, geomDict)
    writeBlockMeshDict(output_folder, geomDict, meshDict)


def generate_blockMeshDict(case_folder):
    input_file = os.path.join(case_folder, "system", "input.json")
    topo_file = os.path.join(case_folder, "system", "topology.json")
    output_folder = os.path.join(case_folder, "system")
    base_mesh(input_file, topo_file, output_folder)


def setupCaseFolder(
    target_folder, geom_template_folder, case_template_folder="case_template"
):
    try:
        rmtree(target_folder)
    except:
        pass
    os.makedirs(target_folder)

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

    copy(os.path.join(case_template_folder, "script_first.sh"), target_folder)
    copy(os.path.join(case_template_folder, "script_second.sh"), target_folder)
    copy(os.path.join(case_template_folder, "Allrun"), target_folder)

    copy(
        os.path.join(geom_template_folder, "input.json"),
        os.path.join(target_folder, "system"),
    )
    copy(
        os.path.join(geom_template_folder, "topology.json"),
        os.path.join(target_folder, "system"),
    )
    generate_blockMeshDict(target_folder)


def gen_slurm_sript(rootFolder):
    f = open(os.path.join(rootFolder, "exec_first.sh"), "w+")
    f.write("for dir in */; do\n")
    f.write("    cd $dir\n")
    f.write("    sbatch script_first.sh\n")
    f.write("    cd ..\n")
    f.write("done\n")
    f.close()


if __name__ == "__main__":
    rootFolder = "widerBCR"
    os.makedirs(rootFolder, exist_ok=True)
    setupCaseFolder(
        os.path.join(rootFolder, "flat_donut"),
        geom_template_folder="flatDonut_widerBCR",
        case_template_folder="case_template_widerBCR",
    )
    setupCaseFolder(
        os.path.join(rootFolder, "circle"),
        geom_template_folder="circle_widerBCR",
        case_template_folder="case_template_widerBCR",
    )
    setupCaseFolder(
        os.path.join(rootFolder, "side_sparger"),
        geom_template_folder="sideSparger_widerBCR",
        case_template_folder="case_template_widerBCR",
    )
    setupCaseFolder(
        os.path.join(rootFolder, "multiring_4"),
        geom_template_folder="multiring_simple4_widerBCR",
        case_template_folder="case_template_widerBCR",
    )
    setupCaseFolder(
        os.path.join(rootFolder, "multiring_2"),
        geom_template_folder="multiring_simple2_widerBCR",
        case_template_folder="case_template_widerBCR",
    )
    gen_slurm_sript(rootFolder)

    rootFolders = [
        "pore_size_1mm",
        "pore_size_2mm",
        "pore_size_3mm",
        "pore_size_4mm",
        "pore_size_5mm",
    ]
    case_template_folders = [
        "case_template_poreSize_1mm",
        "case_template_poreSize_2mm",
        "case_template_poreSize_3mm",
        "case_template_poreSize_4mm",
        "case_template_poreSize_5mm",
    ]

    for rootFolder, case_template_folder in zip(
        rootFolders, case_template_folders
    ):
        os.makedirs(rootFolder, exist_ok=True)
        setupCaseFolder(
            os.path.join(rootFolder, "flat_donut"),
            geom_template_folder="flatDonut",
            case_template_folder=case_template_folder,
        )
        setupCaseFolder(
            os.path.join(rootFolder, "circle"),
            geom_template_folder="circle",
            case_template_folder=case_template_folder,
        )
        setupCaseFolder(
            os.path.join(rootFolder, "side_sparger"),
            geom_template_folder="sideSparger",
            case_template_folder=case_template_folder,
        )
        setupCaseFolder(
            os.path.join(rootFolder, "multiring_4"),
            geom_template_folder="multiring_simple4",
            case_template_folder=case_template_folder,
        )
        setupCaseFolder(
            os.path.join(rootFolder, "multiring_2"),
            geom_template_folder="multiring_simple2",
            case_template_folder=case_template_folder,
        )
        gen_slurm_sript(rootFolder)
