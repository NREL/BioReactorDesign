import sys

import numpy as np
import stl
from scipy.spatial import Delaunay


def triangulate(vertices):
    points = np.zeros((vertices.shape[0], 2))
    points[:, 0] = vertices[:, 0]
    points[:, 1] = vertices[:, 2]
    tri = Delaunay(points)

    return np.array(tri.vertices)


def makePolygon(rad, nvert):
    theta = 2 * np.pi / nvert
    vertices = []
    for i in range(nvert):
        vertices.append(
            [
                rad * np.cos(theta * i + (np.pi / 2 - theta / 2)),
                0,
                rad * np.sin(theta * i + (np.pi / 2 - theta / 2)),
            ]
        )
    vertices = np.array(vertices)
    faces = triangulate(vertices)

    meshInpt = {}
    meshInpt["vertices"] = vertices
    meshInpt["faces"] = faces

    return meshInpt


def makeRectangle(w, h):
    # Define vertices
    vertices = np.array(
        [
            [-w / 2, 0.0, h / 2],
            [w / 2, 0.0, h / 2],
            [w / 2, 0.0, -h / 2],
            [-w / 2, 0.0, -h / 2],
        ]
    )

    faces = triangulate(vertices)

    meshInpt = {}
    meshInpt["vertices"] = vertices
    meshInpt["faces"] = faces

    return meshInpt


def rotate(stlObj, theta=0):
    stlObj.rotate([0, 1, 0], theta)
    return stlObj


def translate(stlObj, vector=np.array([0, 0, 0])):
    stlObj.translate(vector)
    return stlObj


def traceMesh(meshInpt):
    # Create the mesh
    stlObj = stl.mesh.Mesh(np.zeros(meshInpt["faces"].shape[0], dtype=stl.mesh.Mesh.dtype))
    for i, f in enumerate(meshInpt["faces"]):
        for j in range(3):
            stlObj.vectors[i][j] = meshInpt["vertices"][f[j], :]

    return stlObj


def makeSpider(centerRad, nArms, widthArms, lengthArms):
    if nArms < 2:
        print("ERROR: nArms must be >= 2")
        print(f"Got nArms = {nArms}")
        sys.exit()
    if nArms == 2:
        nVertPol = 4
    if nArms > 2:
        nVertPol = nArms
    centerMesh = makePolygon(rad=centerRad, nvert=nVertPol)
    vertices = centerMesh["vertices"]
    maxWidth = np.linalg.norm((vertices[1, :] - vertices[0, :]))
    if widthArms > maxWidth:
        print("ERROR: arm width will make arms overlap")
        print("Either increase center radius or reduce arm width")
        sys.exit()
    center = traceMesh(centerMesh)

    arms = []
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
        arm = traceMesh(makeRectangle(w=widthArms, h=lengthArms))
        side = vertices[indp, :] - vertices[indm, :]
        angle = np.arccos(np.dot(side, [1, 0, 0]) / np.linalg.norm(side))
        if side[2] <= 0:
            angle *= -1
        arm = rotate(arm, angle)
        midSide = (vertices[indp, :] + vertices[indm, :]) / 2
        trans = midSide * (1 + lengthArms / (2 * np.linalg.norm(midSide)))
        arm = translate(arm, trans)
        arms.append(arm)

    arms_data = [entry.data for entry in arms]

    combined = stl.mesh.Mesh(np.concatenate([center.data] + arms_data))

    return combined


def saveSTL(stlObj, filename="spg.stl"):
    # Write the mesh to file
    stlObj.save(filename, mode=stl.Mode.ASCII)
