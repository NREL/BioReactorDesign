# Generate blockmeshDict
python ../../../applications/write_block_rect_mesh.py -i ../../../bird/meshing/block_rect_mesh_templates/loopReactor/input.json -o system

# Generate boundary stl
python ../../../applications/write_stl_patch.py -i ../../../bird/preProcess/stl_patch/bc_patch_mesh_template/loop_reactor/inlets_outlets.json

# Clean case
./Allclean

# Mesh gen
blockMesh -dict system/blockMeshDict

# Inlet BC
surfaceToPatch -tol 1e-3 inlets.stl
export newmeshdir=$(foamListTimes -latestTime)
rm -rf constant/polyMesh/
cp -r $newmeshdir/polyMesh ./constant
rm -rf $newmeshdir
sed -i 's/inlets\.stl/inlet/g' ./constant/polyMesh/boundary

# Outlet BC
surfaceToPatch -tol 1e-3 outlets.stl
export newmeshdir=$(foamListTimes -latestTime)
rm -rf constant/polyMesh/
cp -r $newmeshdir/polyMesh ./constant
rm -rf $newmeshdir
sed -i 's/outlets\.stl/outlet/g' ./constant/polyMesh/boundary


# Scale
transformPoints "scale=(0.05 0.05 0.05)"

# setup IC
cp -r 0.orig 0
setFields

# Run
multiphaseEulerFoam
