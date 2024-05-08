import json

import numpy as np
from stl import mesh


def parseJsonFile(input_filename):
    with open(input_filename) as f:
        inpt = json.load(f)
    return inpt


def make_circle_points(npts, radius, center, normal_dir):
    vertices = np.zeros((npts + 1, 3))
    faces = np.zeros((npts, 3), dtype=int)
    vertices[0, 0] = center[0]
    vertices[0, 1] = center[1]
    vertices[0, 2] = center[2]
    t1dir = (normal_dir + 1) % 3
    t2dir = (normal_dir + 2) % 3
    for i in range(npts):
        ang = i * 2.0 * np.pi / npts
        vertices[i + 1, t1dir] = center[t1dir] + radius * np.cos(ang)
        vertices[i + 1, t2dir] = center[t2dir] + radius * np.sin(ang)
        vertices[i + 1, normal_dir] = center[normal_dir]

        faces[i, 0] = 0
        faces[i, 1] = i % npts + 1
        faces[i, 2] = (i + 1) % npts + 1

    return vertices, faces


def get_all_vert_faces(input_dict, boundary_name):
    offset = int(0)
    boundary_vertices = None
    boundary_faces = None
    for patch in input_dict[boundary_name]:
        if patch["type"].lower() == "circle":
            patch_vertices, patch_faces = make_circle_points(
                patch["nelements"],
                radius=patch["radius"],
                center=(patch["centx"], patch["centy"], patch["centz"]),
                normal_dir=patch["normal_dir"],
            )
            patch_faces += offset
        else:
            raise NotImplementedError
        offset += len(patch_vertices)
        if boundary_vertices is None:
            boundary_vertices = patch_vertices
        else:
            boundary_vertices = np.vstack((boundary_vertices, patch_vertices))
        if boundary_faces is None:
            boundary_faces = patch_faces
        else:
            boundary_faces = np.vstack((boundary_faces, patch_faces))

    return boundary_vertices, boundary_faces

def write_boundaries(input_dict):
    for boundary_name in input_dict.keys():
        boundary_vertices, boundary_faces = get_all_vert_faces(input_dict, boundary_name)
        # Create the mesh
        tri = mesh.Mesh(np.zeros(boundary_faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(boundary_faces):
            for j in range(3):
                tri.vectors[i][j] = boundary_vertices[f[j], :]
        tri.save(f"{boundary_name}.stl")

if __name__ == "__main__":
    input_dict = parseJsonFile(
        "bc_patch_mesh_template/loop_reactor/inlets_outlets.json"
    )
    write_boundaries(input_dict)
