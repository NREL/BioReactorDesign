m4 system/conc_cylinder_mesh.m4 > system/blockMeshDict
rm -rf 0
cp -r 0.org 0
blockMesh
setFields
#decomposePar
