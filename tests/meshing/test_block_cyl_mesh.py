import os
import sys

import numpy as np

from brd import BRD_BLOCK_CYL_MESH_TEMP_DIR
from brd.meshing.block_cyl_mesh import (
    assemble_geom,
    assemble_mesh,
    writeBlockMeshDict,
)


def base_mesh(input_file, topo_file, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    geomDict = assemble_geom(input_file, topo_file)
    meshDict = assemble_mesh(input_file, geomDict)
    writeBlockMeshDict(output_folder, geomDict, meshDict)


def test_side_sparger():
    input_file = os.path.join(
        BRD_BLOCK_CYL_MESH_TEMP_DIR, "sideSparger/input.json"
    )
    topo_file = os.path.join(
        BRD_BLOCK_CYL_MESH_TEMP_DIR, "sideSparger/topology.json"
    )
    output_folder = "system_tmp"
    base_mesh(input_file, topo_file, output_folder)


def test_flat_donut():
    input_file = os.path.join(
        BRD_BLOCK_CYL_MESH_TEMP_DIR, "flatDonut/input.json"
    )
    topo_file = os.path.join(
        BRD_BLOCK_CYL_MESH_TEMP_DIR, "flatDonut/topology.json"
    )
    output_folder = "system_tmp"
    base_mesh(input_file, topo_file, output_folder)


def test_base_column():
    input_file = os.path.join(
        BRD_BLOCK_CYL_MESH_TEMP_DIR, "baseColumn/input.json"
    )
    topo_file = os.path.join(
        BRD_BLOCK_CYL_MESH_TEMP_DIR, "baseColumn/topology.json"
    )
    output_folder = "system_tmp"
    base_mesh(input_file, topo_file, output_folder)


def test_base_column_refine():
    input_file = os.path.join(
        BRD_BLOCK_CYL_MESH_TEMP_DIR, "baseColumn_refineSparg/input.json"
    )
    topo_file = os.path.join(
        BRD_BLOCK_CYL_MESH_TEMP_DIR, "baseColumn_refineSparg/topology.json"
    )
    output_folder = "system_tmp"
    base_mesh(input_file, topo_file, output_folder)


def test_base_column_projected():
    input_file = os.path.join(
        BRD_BLOCK_CYL_MESH_TEMP_DIR, "baseColumn_projected/input.json"
    )
    topo_file = os.path.join(
        BRD_BLOCK_CYL_MESH_TEMP_DIR, "baseColumn_projected/topology.json"
    )
    output_folder = "system_tmp"
    base_mesh(input_file, topo_file, output_folder)


def test_multiring():
    input_file = os.path.join(
        BRD_BLOCK_CYL_MESH_TEMP_DIR, "multiRing_simple/input.json"
    )
    topo_file = os.path.join(
        BRD_BLOCK_CYL_MESH_TEMP_DIR, "multiRing_simple/topology.json"
    )
    output_folder = "system_tmp"
    base_mesh(input_file, topo_file, output_folder)


def test_multiring_coarse():
    input_file = os.path.join(
        BRD_BLOCK_CYL_MESH_TEMP_DIR, "multiRing_coarse/input.json"
    )
    topo_file = os.path.join(
        BRD_BLOCK_CYL_MESH_TEMP_DIR, "multiRing_coarse/topology.json"
    )
    output_folder = "system_tmp"
    base_mesh(input_file, topo_file, output_folder)
