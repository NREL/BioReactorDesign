import json
import sys

import numpy as np
import stl
from bird.meshing.stl_shapes import from_dict_to_stl, create_boundary_patch_list, aggregate_patches 

def parseJsonFile(input_filename):
    with open(input_filename) as f:
        inpt = json.load(f)
    return inpt

def get_all_vert_faces(input_dict, boundary_name):
    patch_mesh_list, patch_area_list = create_boundary_patch_list(input_dict, boundary_name)
    boundary_mesh, boundary_area = aggregate_patches(patch_mesh_list, patch_area_list)
    return boundary_mesh, boundary_area

def write_boundaries(input_dict):
    for boundary_name in input_dict.keys():
        print(f"Making {boundary_name}")
        boundary_mesh, boundary_area = get_all_vert_faces(input_dict, boundary_name)
        tri = from_dict_to_stl(boundary_mesh)
        print(f"\tArea {boundary_area} m2")
        tri.save(f"{boundary_name}.stl", mode=stl.Mode.ASCII)

def saveSTL(stlObj, filename="spg.stl"):
    # Write the mesh to file
    print(f"Generating {filename}")
    stlObj.save(filename, mode=stl.Mode.ASCII)


if __name__ == "__main__":
   input_dict = parseJsonFile(
        "bc_patch_mesh_template/loop_reactor/inlets_outlets.json"
    )
   write_boundaries(input_dict) 
