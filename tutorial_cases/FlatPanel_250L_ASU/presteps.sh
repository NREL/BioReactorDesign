cp -r 0.orig 0
m4 ./system/panel.m4 > ./system/blockMeshDict
blockMesh
setFields
