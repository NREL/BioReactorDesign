import numpy as np
import stl
from stl import mesh
from scipy.spatial import Delaunay

def triangulate(vertices):
    points = np.zeros((vertices.shape[0],2))
    points[:,0] = vertices[:,0]
    points[:,1] = vertices[:,2]
    tri = Delaunay(points)

    return np.array(tri.vertices)

    

def makePolygon(rad, nvert):
    theta = 2*np.pi/nvert
    vertices = []
    for i in range(nvert):
        vertices.append([rad*np.cos(theta*i), 0, rad*np.sin(theta*i)])
    vertices = np.array(vertices)
    faces = triangulate(vertices)   
 
    meshInpt = {}
    meshInpt["vertices"] = vertices
    meshInpt["faces"] = faces

    return meshInpt

def makeCross():
    d = 0.25
    r = d / 2
    w = r / 3
    rs = r / 2
    # Define the 8 vertices of the cube
    vertices = np.array(
        [
            [-w / 2, 0.0, rs],
            [w / 2, 0.0, rs],
            [w / 2, 0.0, w / 2],
            [rs, 0.0, w / 2],
            [rs, 0.0, -w / 2],
            [w / 2, 0.0, -w / 2],
            [w / 2, 0.0, -rs],
            [-w / 2, 0.0, -rs],
            [-w / 2, 0.0, -w / 2],
            [-rs, 0.0, -w / 2],
            [-rs, 0.0, w / 2],
            [-w / 2, 0.0, w / 2],
        ]
    )

    # Define the 12 triangles composing the cube
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 11],
            [2, 3, 4],
            [2, 4, 5],
            [5, 6, 7],
            [5, 7, 8],
            [8, 9, 10],
            [8, 10, 11],
            [8, 11, 5],
            [11, 2, 5],
        ]
    )

    meshInpt = {}
    meshInpt["vertices"] = vertices
    meshInpt["faces"] = faces

    return meshInpt


def makeRectangle(w,h, shiftw=0, shifth=0):
    # Define the 8 vertices of the cube
    vertices = np.array(
        [
            [-w / 2, 0.0, h/2],
            [w / 2, 0.0, h/2],
            [w / 2, 0.0, -h/2],
            [-w / 2, 0.0, -h/2] 
        ]
    )
    vertices[:,0] += shiftw 
    vertices[:,2] += shifth

    # Define the 12 triangles composing the cube
    faces = np.array(
        [
            [0, 1, 2],
            [2, 3, 0],
        ]
    )

    meshInpt = {}
    meshInpt["vertices"] = vertices
    meshInpt["faces"] = faces

    return meshInpt


def traceMesh(meshInpt):
    # Create the mesh
    stlObj = mesh.Mesh(np.zeros(meshInpt["faces"].shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(meshInpt["faces"]):
        for j in range(3):
            stlObj.vectors[i][j] = meshInpt["vertices"][f[j], :]

    return stlObj


def saveSTL(stlObj):
    # Write the mesh to file "cube.stl"
    stlObj.save("spg.stl", mode=stl.Mode.ASCII)
