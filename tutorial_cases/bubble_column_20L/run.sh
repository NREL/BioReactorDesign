#!/bin/bash

# Clean case
./Allclean

set -e  # Exit on any error
# Define what to do on error
trap 'echo "ERROR: Something failed! Running cleanup..."; ./Allclean' ERR


echo PRESTEP 1
# Generate blockmeshDict
python ../../applications/write_block_cyl_mesh.py -i system/mesh.json -t system/topology.json -o system

# Generate boundary stl
python ../../applications/write_stl_patch.py -i system/inlets_outlets.json

echo PRESTEP 2
# Mesh gen
blockMesh -dict system/blockMeshDict
transformPoints "rotate=((0 0 1) (0 1 0))"
transformPoints "scale=(0.001 0.001 0.001)"

# Inlet BC
surfaceToPatch -tol 1e-3 inlets.stl
export newmeshdir=$(foamListTimes -latestTime)
rm -rf constant/polyMesh/
cp -r $newmeshdir/polyMesh ./constant
rm -rf $newmeshdir
cp constant/polyMesh/boundary /tmp
sed -i -e 's/inlets\.stl/inlet/g' /tmp/boundary
cat /tmp/boundary > constant/polyMesh/boundary

# setup IC
cp -r 0.orig 0
setFields

# Scale
transformPoints "scale=(0.19145161188225573 0.19145161188225573 0.19145161188225573)"

# Setup mass flow rate
# Get inlet area
postProcess -func 'patchIntegrate(patch="inlet", field="alpha.gas")'
postProcess -func writeCellVolumes
writeMeshObj

echo PRESTEP 3
python writeGlobalVars.py
cp constant/phaseProperties_pbe constant/phaseProperties


echo RUN
birdmultiphaseEulerFoam
