import numpy as np
from stl import mesh
from sys import argv

length=12*0.025 #12 inches
width=1.0*0.025 #1 inch

vertices=np.zeros((5,3));
faces=np.zeros((4,3))

vertices[0,0]=-0.5*length
vertices[0,1]=0.0
vertices[0,2]=-0.5*width

vertices[1,0]=0.5*length
vertices[1,1]=0.0
vertices[1,2]=-0.5*width

vertices[2,0]=0.5*length
vertices[2,1]=0.0
vertices[2,2]=0.5*width

vertices[3,0]=-0.5*length
vertices[3,1]=0.0
vertices[3,2]=0.5*width

vertices[4,0]=0.0
vertices[4,1]=0.0
vertices[4,2]=0.0

faces[0,0]=0;
faces[0,1]=1;
faces[0,2]=4;

faces[1,0]=1;
faces[1,1]=2;
faces[1,2]=4;

faces[2,0]=2;
faces[2,1]=3;
faces[2,2]=4;

faces[3,0]=3;
faces[3,1]=0;
faces[3,2]=4;

faces=faces.astype(int)
# Create the mesh
cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        cube.vectors[i][j] = vertices[f[j],:]

# Write the mesh to file "cube.stl"
cube.save('sparger.stl')
