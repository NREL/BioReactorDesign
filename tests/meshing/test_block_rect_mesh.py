import os
import sys

import numpy as np

from bird import BIRD_BLOCK_RECT_MESH_TEMP_DIR
from bird.meshing.block_rect_mesh import (
    assemble_geom,
    assemble_mesh,
    writeBlockMeshDict,
)


def base_mesh(input_file, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    geomDict = assemble_geom(input_file)
    meshDict = assemble_mesh(input_file)
    writeBlockMeshDict(output_folder, geomDict, meshDict)


def test_loop_reactor():
    input_file = os.path.join(
        BIRD_BLOCK_RECT_MESH_TEMP_DIR, "loopReactor/input.json"
    )
    output_folder = "system_tmp"
    base_mesh(input_file, output_folder)
