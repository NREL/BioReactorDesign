# Clean case
./Allclean

# Generate blockmeshDict
python ../../../applications/write_block_cyl_mesh.py -i ../../../bird/meshing/block_cyl_mesh_templates/sideSparger/input.json  -t ../../../bird/meshing/block_cyl_mesh_templates/sideSparger/topology.json -o system


# Mesh gen
blockMesh -dict system/blockMeshDict
transformPoints "scale=(0.001 0.001 0.001)"
transformPoints "rotate=((0 0 1) (0 1 0))"

# Set IC
cp -r 0.orig 0
setFields

# Setup files for paraview
touch sol.foam

# Write mesh details for post processing
writeMeshObj

# Run
multiphaseEulerFoam



