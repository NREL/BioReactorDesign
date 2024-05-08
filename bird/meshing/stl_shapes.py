import json
import sys

import numpy as np
import stl
from scipy.spatial import Delaunay

def from_dict_to_stl(meshInpt):
    # Create the mesh
    stlObj = stl.mesh.Mesh(
        np.zeros(meshInpt["faces"].shape[0], dtype=stl.mesh.Mesh.dtype)
    )
    for i, f in enumerate(meshInpt["faces"]):
        for j in range(3):
            stlObj.vectors[i][j] = meshInpt["vertices"][f[j], :]
    return stlObj



def from_stl_to_dict(stlObj):
    vertices = np.unique(np.reshape(stlObj.vectors, (-1,3)), axis=0)
    faces = np.zeros((stlObj.vectors.shape[0], 3), dtype=int)
    for i in range(stlObj.vectors.shape[0]):
        for j in range(stlObj.vectors.shape[1]):
            ind = np.argwhere(np.linalg.norm(vertices-stlObj.vectors[i,j], axis=1)==0)[0][0]
            faces[i,j] = ind
    return {"vertices": vertices, "faces": faces}

def tri_area(point1, point2, point3):
    return 0.5 * (
        point1[0] * (point2[1] - point3[1])
        + point2[0] * (point3[1] - point1[1])
        + point3[0] * (point1[1] - point2[1])
    )


def stl_area(tri):
    vertices = tri.simplices
    points = tri.points
    area = 0
    for i_triangle in range(vertices.shape[0]):
        area += tri_area(
            points[vertices[i_triangle, 0], :],
            points[vertices[i_triangle, 1], :],
            points[vertices[i_triangle, 2], :],
        )
    return area

def aggregate_patches(patch_mesh_list, patch_area_list):
    offset = int(0)
    boundary_area = 0
    boundary_mesh = {}
    boundary_mesh["vertices"] = None
    boundary_mesh["faces"] = None
    for (patch_mesh, patch_area) in zip(patch_mesh_list, patch_area_list):
        patch_mesh["faces"] += offset
        offset += len(patch_mesh["vertices"])
        if boundary_mesh["vertices"] is None:
            boundary_mesh["vertices"] = patch_mesh["vertices"]
        else:
            boundary_mesh["vertices"] = np.vstack((boundary_mesh["vertices"], patch_mesh["vertices"]))
        if boundary_mesh["faces"] is None:
            boundary_mesh["faces"] = patch_mesh["faces"]
        else:
            boundary_mesh["faces"] = np.vstack((boundary_mesh["faces"], patch_mesh["faces"]))
        boundary_area += patch_area
    return boundary_mesh, boundary_area



def triangulate(vertices):
    points = np.zeros((vertices.shape[0], 2))
    points[:, 0] = vertices[:, 0]
    points[:, 1] = vertices[:, 2]
    tri = Delaunay(points)
    area = stl_area(tri)
    return np.array(tri.simplices), area

def rotate_stl(stlObj, theta=0, normal_dir=1):
    normal = [0, 0, 0]
    normal[normal_dir] = 1
    stlObj.rotate(normal, theta)
    return stlObj

def rotate_mesh_dict(mesh_dict, theta=0, normal_dir=1):
    #mesh = from_dict_to_stl(mesh_dict)
    #normal = [0, 0, 0]
    #normal[normal_dir] = 1
    #mesh.rotate(normal, theta)
    #return from_stl_to_dict(mesh)
    rot_mat = np.zeros((3,3))
    
    if normal_dir == 0:
        rot_mat[0,0] = 1
        rot_mat[0,1] = 0
        rot_mat[0,2] = 0
        rot_mat[1,0] = 0
        rot_mat[1,1] = np.cos(theta)
        rot_mat[1,2] = -np.sin(theta)
        rot_mat[2,0] = 0
        rot_mat[2,1] = np.sin(theta)
        rot_mat[2,2] = np.cos(theta)
    elif normal_dir == 1:
        rot_mat[0,0] = np.cos(theta)
        rot_mat[0,1] = 0
        rot_mat[0,2] = np.sin(theta)
        rot_mat[1,0] = 0
        rot_mat[1,1] = 1
        rot_mat[1,2] = 0
        rot_mat[2,0] = -np.sin(theta)
        rot_mat[2,1] = 0
        rot_mat[2,2] = np.cos(theta)
    elif normal_dir == 2:
        rot_mat[0,0] = np.cos(theta)
        rot_mat[0,1] = -np.sin(theta)
        rot_mat[0,2] = 0
        rot_mat[1,0] = np.sin(theta)
        rot_mat[1,1] = np.cos(theta)
        rot_mat[1,2] = 0
        rot_mat[2,0] = 0
        rot_mat[2,1] = 0
        rot_mat[2,2] = 1

    for i in range(mesh_dict["vertices"].shape[0]):
        tmp = rot_mat @ mesh_dict["vertices"][i]
        mesh_dict["vertices"][i] = tmp

    return mesh_dict

def translate_stl(stlObj, vector=np.array([0, 0, 0])):
    stlObj.translate(vector)
    return stlObj

def translate_mesh_dict(mesh_dict, vector=np.array([0, 0, 0])):
    mesh_dict["vertices"] += vector
    return mesh_dict


def make_polygon(rad, nvert, center, normal_dir):
    theta = 2 * np.pi / nvert
    vertices = np.zeros((nvert, 3))
    t1dir = (normal_dir + 1) % 3
    t2dir = (normal_dir + 2) % 3
    for i in range(nvert):
        vertices[i, t2dir] = rad * np.cos(theta * i + (np.pi / 2 - theta / 2))
        vertices[i, t1dir] = rad * np.sin(theta * i + (np.pi / 2 - theta / 2))
    
    vertices = np.array(vertices)
    faces, area = triangulate(vertices)

    mesh_dict = {}
    mesh_dict["vertices"] = vertices
    mesh_dict["faces"] = faces
 
    mesh_dict = translate_mesh_dict(mesh_dict, center)

    return mesh_dict, area


def make_rectangle(w, h, center, normal_dir):
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

    faces, area = triangulate(vertices)

    mesh_dict = {}
    mesh_dict["vertices"] = vertices
    mesh_dict["faces"] = faces

    mesh_dict = translate_mesh_dict(mesh_dict, center)

    return mesh_dict, area

def make_circle(radius, center, normal_dir, npts=3):
    vertices = np.zeros((npts + 1, 3))
    t1dir = (normal_dir + 1) % 3
    t2dir = (normal_dir + 2) % 3
    for i in range(npts):
        ang = i * 2.0 * np.pi / npts
        vertices[i + 1, t2dir] = radius * np.cos(ang)
        vertices[i + 1, t1dir] = radius * np.sin(ang)
  
    faces, area = triangulate(vertices)

    mesh_dict = {}
    mesh_dict["vertices"] = vertices
    mesh_dict["faces"] = faces
 

    mesh_dict = translate_mesh_dict(mesh_dict, center)

    return mesh_dict, area


def make_spider(centerRad, nArms, widthArms, lengthArms, center, normal_dir):
    #print("Making spider")
    #print(f"\tcenterRadius={centerRad}")
    #print(f"\tnArms={nArms}")
    #print(f"\twidthArms={widthArms}")
    #print(f"\tlengthArms={lengthArms}")
   
    globalArea = 0
    if nArms < 2:
        print("ERROR: nArms must be >= 2")
        print(f"Got nArms = {nArms}")
        sys.exit()
    if nArms == 2:
        nVertPol = 4
    if nArms > 2:
        nVertPol = nArms
    centerMesh, centerArea = make_polygon(rad=centerRad, nvert=nVertPol, center=(0,0,0), normal_dir=normal_dir)
    globalArea += centerArea
    vertices = centerMesh["vertices"]
    maxWidth = np.linalg.norm((vertices[1, :] - vertices[0, :]))
    if widthArms > maxWidth:
        print("ERROR: arm width will make arms overlap")
        print("Either increase center radius or reduce arm width")
        sys.exit()

    arm_mesh_list = []
    arm_area_list = []
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
        armMesh, armArea = make_rectangle(w=widthArms, h=lengthArms, center=(0,0,0), normal_dir=normal_dir)
        globalArea += armArea
        side = vertices[indp, :] - vertices[indm, :]
        angle = np.arccos(np.dot(side, [1, 0, 0]) / np.linalg.norm(side))
        if side[2] <= 0:
            angle *= -1
        armMesh = rotate_mesh_dict(armMesh, angle)
        midSide = (vertices[indp, :] + vertices[indm, :]) / 2
        trans = midSide * (1 + lengthArms / (2 * np.linalg.norm(midSide)))
        armMesh = translate_mesh_dict(armMesh, trans)
        arm_mesh_list.append(armMesh)
        arm_area_list.append(armArea)

    mesh_dict, area = aggregate_patches([centerMesh] + arm_mesh_list, [centerArea] + arm_area_list)
    mesh_dict = translate_mesh_dict(mesh_dict, center)

    return mesh_dict, area

def create_boundary_patch_list(input_dict, boundary_name):
    patch_mesh_list = []
    patch_area_list = []
    for patch in input_dict[boundary_name]:
        if patch["type"].lower() == "circle":
            patch_mesh, patch_area = make_circle(
                radius=patch["radius"],
                center=(patch["centx"], patch["centy"], patch["centz"]),
                normal_dir=patch["normal_dir"],
                npts=patch["nelements"],
            )
        elif patch["type"].lower() == "spider":
            patch_mesh, patch_area = make_spider(
                centerRad=patch["centerRad"],
                nArms=patch["nArms"],
                widthArms=patch["widthArms"],
                lengthArms=patch["lengthArms"],
                center=(patch["centx"], patch["centy"], patch["centz"]),
                normal_dir=patch["normal_dir"],
            )
        elif patch["type"].lower() == "polygon": 
            patch_mesh, patch_area = make_polygon(
                rad=patch["radius"],
                nvert=patch["npts"],
            )
        elif patch["type"].lower() == "rectangle":
            patch_mesh, patch_area = make_rectangle(
                w=patch["width"],
                h=patch["height"],
                center=(patch["centx"], patch["centy"], patch["centz"]),
                normal_dir=patch["normal_dir"],
            )
        else:
            raise NotImplementedError
        patch_mesh_list.append(patch_mesh)
        patch_area_list.append(patch_area)

    return patch_mesh_list, patch_area_list
 
