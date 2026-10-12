#!/bin/bash

# Clean case
./Allclean

set -Eeuo pipefail   # -E: trap also fires inside functions/subshells
trap 'exit_code=$?
      echo "ERROR: Something failed! Running cleanup..."
      ./Allclean || true
      exit $exit_code' ERR


# Generate blockmeshDict
python ../../../applications/write_block_rect_mesh.py -i system/mesh.json -o system

# Generate boundary stl
python ../../../applications/write_stl_patch.py -i system/inlets_outlets.json

# Generate species thermo properties
python ../../../applications/write_species_thermo_prop.py -cf .
    
# Mesh gen
blockMesh -dict system/blockMeshDict

# Inlet BC
surfaceToPatch -tol 1e-3 inlets.stl
newmeshdir=$(foamListTimes -latestTime)
export newmeshdir
rm -rf constant/polyMesh/
cp -r $newmeshdir/polyMesh ./constant
rm -rf $newmeshdir
cp constant/polyMesh/boundary /tmp
sed -i -e 's/inlets\.stl/inlet/g' /tmp/boundary
cat /tmp/boundary > constant/polyMesh/boundary

# Outlet BC
surfaceToPatch -tol 1e-3 outlets.stl
newmeshdir=$(foamListTimes -latestTime)
export newmeshdir
rm -rf constant/polyMesh/
cp -r $newmeshdir/polyMesh ./constant
rm -rf $newmeshdir
cp constant/polyMesh/boundary /tmp
sed -i -e 's/outlets\.stl/outlet/g' /tmp/boundary
cat /tmp/boundary > constant/polyMesh/boundary

# Scale
transformPoints "scale=(2.7615275385627096 2.7615275385627096 2.7615275385627096)"

# setup IC
cp -r 0.orig 0
setFields

# Setup mass flow rate
# Get inlet area
postProcess -func 'patchIntegrate(patch="inlet", field="alpha.gas")'
postProcess -func writeCellVolumes
writeMeshObj

echo PRESTEP 3
python writeGlobalVars.py
cp constant/phaseProperties_pbe constant/phaseProperties

# Run
birdmultiphaseEulerFoam




