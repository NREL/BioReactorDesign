import os
import sys

import numpy as np

sys.path.append("../util")
import argument
from meshing import *
from myparser import parseJsonFile

sys.path.append("..")
from writeBlockMesh import *


def base_mesh(argsDict):
    geomDict = assemble_geom(argsDict)
    meshDict = assemble_mesh(argsDict, geomDict)
    writeBlockMeshDict(argsDict, geomDict, meshDict)

def test_side_sparger():
    argsDict = {
        "input_file": "../sideSparger/input.json",
        "topo_file": "../sideSparger/topology.json",
        "out_folder": "../case/system",
    }
    base_mesh(argsDict)


def test_flat_donut():
    argsDict = {
        "input_file": "../flatDonut/input.json",
        "topo_file": "../flatDonut/topology.json",
        "out_folder": "../case/system",
    }
    base_mesh(argsDict)

def test_base_column():
    argsDict = {
        "input_file": "../baseColumn/input.json",
        "topo_file": "../baseColumn/topology.json",
        "out_folder": "../case/system",
    }
    base_mesh(argsDict)
