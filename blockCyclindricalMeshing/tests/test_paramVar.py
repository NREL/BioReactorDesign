import os
import sys

import numpy as np

sys.path.append("../util")
import argument
from meshing import *
from modifyGeom import *
from myparser import parseJsonFile

sys.path.append("..")
from distutils.dir_util import copy_tree
from shutil import copy, rmtree

from sparger_optim import *
from writeBlockMesh import *


def test_multiRing_variations():
    study_folder = "../study"
    try:
        rmtree(study_folder)
    except:
        pass
    multi_ring_variations(
        10,
        study_folder=study_folder,
        case_template_folder="../case_template",
        template_folder="../template_multiRing",
    )


def test_multiRing_num_variations():
    study_folder = "../study"
    try:
        rmtree(study_folder)
    except:
        pass
    multi_ring_num_variations(
        study_folder=study_folder,
        template_root_folder="..",
        case_template_folder="../case_template",
    )


def test_flatDonut_variations():
    study_folder = "../study"
    try:
        rmtree(study_folder)
    except:
        pass
    flat_donut_variations(
        10,
        study_folder=study_folder,
        case_template_folder="../case_template",
        template_folder="../template_flatDonut",
    )


def test_sideSparger_variations():
    study_folder = "../study"
    try:
        rmtree(study_folder)
    except:
        pass
    side_sparger_variations(
        10,
        study_folder=study_folder,
        case_template_folder="../case_template",
        template_folder="../template_sideSparger",
    )
