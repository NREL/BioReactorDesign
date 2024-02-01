import sys

import numpy as np
import stl
from scipy.spatial import Delaunay


def tri_area(point1, point2, point3):
    return 0.5 * (
        point1[0] * (point2[1] - point3[1])
        + point2[0] * (point3[1] - point1[1])
        + point3[0] * (point1[1] - point2[1])
    )


def patch_area(tri):
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


def triangulate(vertices):
    points = np.zeros((vertices.shape[0], 2))
    points[:, 0] = vertices[:, 0]
    points[:, 1] = vertices[:, 2]
    tri = Delaunay(points)
    area = patch_area(tri)
    return np.array(tri.simplices), area


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
    faces, area = triangulate(vertices)

    meshInpt = {}
    meshInpt["vertices"] = vertices
    meshInpt["faces"] = faces

    return meshInpt, area


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

    faces, area = triangulate(vertices)

    meshInpt = {}
    meshInpt["vertices"] = vertices
    meshInpt["faces"] = faces

    return meshInpt, area


def rotate(stlObj, theta=0):
    stlObj.rotate([0, 1, 0], theta)
    return stlObj


def translate(stlObj, vector=np.array([0, 0, 0])):
    stlObj.translate(vector)
    return stlObj


def traceMesh(meshInpt):
    # Create the mesh
    stlObj = stl.mesh.Mesh(
        np.zeros(meshInpt["faces"].shape[0], dtype=stl.mesh.Mesh.dtype)
    )
    for i, f in enumerate(meshInpt["faces"]):
        for j in range(3):
            stlObj.vectors[i][j] = meshInpt["vertices"][f[j], :]

    return stlObj


def makeSpider(centerRad, nArms, widthArms, lengthArms):
    print("Making spider")
    print(f"\tcenterRadius={centerRad}")
    print(f"\tnArms={nArms}")
    print(f"\twidthArms={widthArms}")
    print(f"\tlengthArms={lengthArms}")

    globalArea = 0
    if nArms < 2:
        print("ERROR: nArms must be >= 2")
        print(f"Got nArms = {nArms}")
        sys.exit()
    if nArms == 2:
        nVertPol = 4
    if nArms > 2:
        nVertPol = nArms
    centerMesh, centerArea = makePolygon(rad=centerRad, nvert=nVertPol)
    globalArea += centerArea
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
        armMesh, armArea = makeRectangle(w=widthArms, h=lengthArms)
        globalArea += armArea
        arm = traceMesh(armMesh)
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

    return combined, globalArea


def saveSTL(stlObj, filename="spg.stl"):
    # Write the mesh to file
    print(f"Generating {filename}")
    stlObj.save(filename, mode=stl.Mode.ASCII)
