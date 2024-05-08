import json
import sys

import numpy as np
import stl

from bird.preProcess.stl_patch.stl_shapes import *


def parseJsonFile(input_filename):
    with open(input_filename) as f:
        inpt = json.load(f)
    return inpt


def check_input(input_dict):
    assert isinstance(input_dict, dict)
    for bound in input_dict:
        assert isinstance(input_dict[bound], list)
        for patch in input_dict[bound]:
            assert isinstance(patch, dict)


def get_all_vert_faces(input_dict, boundary_name):
    patch_mesh_list = create_boundary_patch_list(input_dict, boundary_name)
    boundary_mesh = STLMesh()
    boundary_mesh.from_mesh_list(patch_mesh_list)
    return boundary_mesh


def write_boundaries(input_dict):
    check_input(input_dict)
    for boundary_name in input_dict.keys():
        print(f"Making {boundary_name}")
        boundary_mesh = get_all_vert_faces(input_dict, boundary_name)
        print(f"\tArea {boundary_mesh.area} m2")
        boundary_mesh.save(f"{boundary_name}.stl")


if __name__ == "__main__":
    input_dict = parseJsonFile(
        "bc_patch_mesh_template/loop_reactor/inlets_outlets.json"
    )
    write_boundaries(input_dict)
