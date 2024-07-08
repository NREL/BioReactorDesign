rm -rf 0
cp -r 0.org 0
blockMesh
snappyHexMesh -overwrite
setFields
decomposePar
