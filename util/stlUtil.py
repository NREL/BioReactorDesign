import numpy as np
import stl
from stl import mesh

def makeCross():

    d=0.25
    r=d/2
    w=r/3
    rs=r/2
    # Define the 8 vertices of the cube
    vertices = np.array([\
        [-w/2, 0.0, rs],
        [w/2,  0.0, rs],
        [w/2,  0.0, w/2],
        [rs,   0.0, w/2],
        [rs,   0.0, -w/2],
        [w/2,  0.0,-w/2],
        [w/2,   0.0, -rs],
        [-w/2,   0.0, -rs],
        [-w/2,   0.0, -w/2],
        [-rs,   0.0, -w/2],
        [-rs,   0.0, w/2],
        [-w/2, 0.0, w/2]])
    
        # Define the 12 triangles composing the cube
    faces = np.array([\
        [0,1,2],
        [0,2,11],
        [2,3,4],
        [2,4,5],
        [5,6,7],
        [5,7,8],
        [8,9,10],
        [8,10,11],
        [8,11,5],
        [11,2,5]])
 
    meshInpt = {}
    meshInpt["vertices"] = vertices
    meshInpt["faces"] = faces

    return meshInpt


def traceMesh(meshInpt):
    # Create the mesh
    stlObj = mesh.Mesh(np.zeros(meshInpt["faces"].shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(meshInpt["faces"]):
        for j in range(3):
            stlObj.vectors[i][j] = meshInpt["vertices"][f[j],:]

    return stlObj


def saveSTL(stlObj):
    # Write the mesh to file "cube.stl"
    stlObj.save('spg.stl',mode=stl.Mode.ASCII)
