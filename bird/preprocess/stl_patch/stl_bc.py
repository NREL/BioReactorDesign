import os

import numpy as np
import stl

from bird import logger
from bird.preprocess.stl_patch.stl_shapes import *
from bird.utilities.parser import parse_json


def check_input(input_dict):
    assert isinstance(input_dict, dict)
    need_geom = False
    for bound in input_dict:
        if not bound == "Geometry" and not bound == "Meshing":
            assert isinstance(input_dict[bound], list)
            for patch in input_dict[bound]:
                assert isinstance(patch, dict)
                if "branch_id" in patch:
                    need_geom = True
    if need_geom:
        assert "Geometry" in input_dict
        assert "OverallDomain" in input_dict["Geometry"]
        assert "x" in input_dict["Geometry"]["OverallDomain"]
        assert "y" in input_dict["Geometry"]["OverallDomain"]
        assert "z" in input_dict["Geometry"]["OverallDomain"]
        assert "size_per_block" in input_dict["Geometry"]["OverallDomain"]["x"]
        assert "Fluids" in input_dict["Geometry"]
        assert isinstance(input_dict["Geometry"]["Fluids"], list)
        assert isinstance(input_dict["Geometry"]["Fluids"][0], list)


def get_all_vert_faces(input_dict, boundary_name):
    patch_mesh_list = create_boundary_patch_list(input_dict, boundary_name)
    boundary_mesh = STLMesh()
    boundary_mesh.from_mesh_list(patch_mesh_list)
    return boundary_mesh


def write_boundaries(input_dict, output_folder="."):
    check_input(input_dict)
    for boundary_name in input_dict.keys():
        if not boundary_name == "Geometry":
            logger.info(f"Making {boundary_name}")
            boundary_mesh = get_all_vert_faces(input_dict, boundary_name)
            logger.info(f"\tArea {boundary_mesh.area} m2")
            boundary_mesh.save(
                os.path.join(output_folder, f"{boundary_name}.stl")
            )


if __name__ == "__main__":
    input_dict = parse_json(
        "bc_patch_mesh_template/loop_reactor_expl/inlets_outlets.json"
    )
    write_boundaries(input_dict)
    input_dict = parse_json(
        "bc_patch_mesh_template/loop_reactor_branch/inlets_outlets.json"
    )
    write_boundaries(input_dict)
