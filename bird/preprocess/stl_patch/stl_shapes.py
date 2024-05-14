import json
import sys

import numpy as np

from bird.preprocess.stl_patch.stl_mesh import STLMesh


def make_polygon(rad, nvert, center, normal_dir):
    print("\tMaking polygon")
    theta = 2 * np.pi / nvert
    vertices = np.zeros((nvert, 3))
    t1dir = (normal_dir + 1) % 3
    t2dir = (normal_dir + 2) % 3
    for i in range(nvert):
        vertices[i, t2dir] = rad * np.cos(theta * i + (np.pi / 2 - theta / 2))
        vertices[i, t1dir] = rad * np.sin(theta * i + (np.pi / 2 - theta / 2))

    stl_mesh = STLMesh(vertices=vertices, planar=True, normal_dir=normal_dir)
    stl_mesh.translate(center)

    return stl_mesh


def make_rectangle(w, h, center, normal_dir):
    print("\tMaking rectangle")
    # Define vertices
    t1dir = (normal_dir + 1) % 3
    t2dir = (normal_dir + 2) % 3
    vertices = np.zeros((4, 3))
    vertices[0, t2dir] = -w / 2
    vertices[0, t1dir] = h / 2
    vertices[1, t2dir] = w / 2
    vertices[1, t1dir] = h / 2
    vertices[2, t2dir] = w / 2
    vertices[2, t1dir] = -h / 2
    vertices[3, t2dir] = -w / 2
    vertices[3, t1dir] = -h / 2

    stl_mesh = STLMesh(vertices=vertices, planar=True, normal_dir=normal_dir)
    stl_mesh.translate(center)

    return stl_mesh


def make_circle(radius, center, normal_dir, npts=3):
    print("\tMaking circle")
    vertices = np.zeros((npts + 1, 3))
    t1dir = (normal_dir + 1) % 3
    t2dir = (normal_dir + 2) % 3
    for i in range(npts):
        ang = i * 2.0 * np.pi / npts
        vertices[i + 1, t2dir] = radius * np.cos(ang)
        vertices[i + 1, t1dir] = radius * np.sin(ang)

    stl_mesh = STLMesh(vertices=vertices, planar=True, normal_dir=normal_dir)
    stl_mesh.translate(center)

    return stl_mesh


def make_spider(centerRad, nArms, widthArms, lengthArms, center, normal_dir):
    globalArea = 0
    if nArms < 2:
        print("ERROR: nArms must be >= 2")
        print(f"Got nArms = {nArms}")
        sys.exit()
    if nArms == 2:
        nVertPol = 4
    if nArms > 2:
        nVertPol = nArms
    center_mesh = make_polygon(
        rad=centerRad, nvert=nVertPol, center=(0, 0, 0), normal_dir=normal_dir
    )
    vertices = center_mesh.vertices
    maxWidth = np.linalg.norm((vertices[1, :] - vertices[0, :]))
    if widthArms > maxWidth:
        print("ERROR: arm width will make arms overlap")
        print("Either increase center radius or reduce arm width")
        sys.exit()

    arm_mesh_list = []
    for i in range(nArms):
        if nArms > 2:
            if i < nArms - 1:
                indp = i + 1
                indm = i
            else:
                indp = 0
                indm = i
        if nArms == 2:
            if i == 0:
                indp = 1
                indm = 0
            else:
                indp = 3
                indm = 2
        arm_mesh = make_rectangle(
            w=widthArms, h=lengthArms, center=(0, 0, 0), normal_dir=normal_dir
        )
        side = vertices[indp, :] - vertices[indm, :]
        if normal_dir == 1:
            angle = np.arccos(np.dot(side, [1, 0, 0]) / np.linalg.norm(side))
            if side[2] <= 0:
                angle *= -1
        else:
            raise NotImplementedError
        arm_mesh.rotate(angle, normal_dir=normal_dir)
        midSide = (vertices[indp, :] + vertices[indm, :]) / 2
        trans = midSide * (1 + lengthArms / (2 * np.linalg.norm(midSide)))
        arm_mesh.translate(trans)
        arm_mesh_list.append(arm_mesh)

    spider_mesh = STLMesh()
    spider_mesh.from_mesh_list([center_mesh] + arm_mesh_list)
    spider_mesh.translate(center)
    spider_mesh.planar = True
    spider_mesh.normal_dir = normal_dir

    return spider_mesh


def create_boundary_patch_list(input_dict, boundary_name):
    patch_mesh_list = []
    for patch in input_dict[boundary_name]:
        if patch["type"].lower() == "circle":
            patch_mesh = make_circle(
                radius=patch["radius"],
                center=(patch["centx"], patch["centy"], patch["centz"]),
                normal_dir=patch["normal_dir"],
                npts=patch["nelements"],
            )
        elif patch["type"].lower() == "spider":
            patch_mesh = make_spider(
                centerRad=patch["centerRad"],
                nArms=patch["nArms"],
                widthArms=patch["widthArms"],
                lengthArms=patch["lengthArms"],
                center=(patch["centx"], patch["centy"], patch["centz"]),
                normal_dir=patch["normal_dir"],
            )
        elif patch["type"].lower() == "polygon":
            patch_mesh = make_polygon(
                rad=patch["radius"],
                nvert=patch["npts"],
                center=(patch["centx"], patch["centy"], patch["centz"]),
                normal_dir=patch["normal_dir"],
            )
        elif patch["type"].lower() == "rectangle":
            patch_mesh = make_rectangle(
                w=patch["width"],
                h=patch["height"],
                center=(patch["centx"], patch["centy"], patch["centz"]),
                normal_dir=patch["normal_dir"],
            )
        else:
            raise NotImplementedError
        patch_mesh_list.append(patch_mesh)

    return patch_mesh_list
