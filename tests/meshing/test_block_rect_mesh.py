import os
import sys
import tempfile
from pathlib import Path

import numpy as np

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
    BIRD_BLOCK_RECT_MESH_TEMP_DIR = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "bird",
        "meshing",
        "block_rect_mesh_templates",
    )
    input_file = os.path.join(
        BIRD_BLOCK_RECT_MESH_TEMP_DIR, "loopReactor", "input.json"
    )
    # Output to temporary directory and delete when done
    with tempfile.TemporaryDirectory() as tmpdirname:
        output_folder = tmpdirname
        base_mesh(input_file, output_folder)
        assert os.path.exists(os.path.join(tmpdirname, "blockMeshDict"))


def test_subblock_reactor():
    BIRD_BLOCK_RECT_MESH_TEMP_DIR = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "bird",
        "meshing",
        "block_rect_mesh_templates",
    )
    input_file = os.path.join(
        BIRD_BLOCK_RECT_MESH_TEMP_DIR, "sub_blocks", "input.json"
    )
    # Output to temporary directory and delete when done
    with tempfile.TemporaryDirectory() as tmpdirname:
        output_folder = tmpdirname
        base_mesh(input_file, output_folder)
        assert os.path.exists(os.path.join(tmpdirname, "blockMeshDict"))
