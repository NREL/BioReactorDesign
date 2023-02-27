import numpy as np
import stl
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def plotSTL(mesh):
    fig = plt.figure() 
    axes = mplot3d.Axes3D(fig)
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh.vectors))
    eps = 1e-16
    axes.set_xlim3d(left=np.amin(mesh.vectors[:,:,0])-eps, right=np.amax(mesh.vectors[:,:,0])+eps)
    axes.set_ylim3d(np.amin(mesh.vectors[:,:,1])-eps, np.amax(mesh.vectors[:,:,1])+eps)
    axes.set_zlim3d(np.amin(mesh.vectors[:,:,2])-eps,np.amax(mesh.vectors[:,:,2])+eps)
    axes.view_init(0,90) 
    plt.yticks([])
    plt.show()


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

# Create the mesh
cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        cube.vectors[i][j] = vertices[f[j],:]

# Write the mesh to file "cube.stl"
cube.save('spg.stl',mode=stl.Mode.ASCII)

plotSTL(cube)
