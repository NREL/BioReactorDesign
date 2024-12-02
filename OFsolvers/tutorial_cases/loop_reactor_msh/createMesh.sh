# !/bib/bash

gmsh -3 UTube.geo -o UTube.stl
#mv UTube.stl constant/triSurface
#surfaceFeatures
#blockMesh
#snappyHexMesh
gmshToFoam UTube.msh
#checkMesh -allGeometry -allTopology
#polyDualMesh -overwrite -concaveMultiCells  0
checkMesh -allGeometry -allTopology



