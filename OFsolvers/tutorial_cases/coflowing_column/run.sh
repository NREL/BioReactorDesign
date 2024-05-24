# Clean case
./Allclean

# Generate blockmeshDict
python ../../../applications/write_block_cyl_mesh.py -i ../../../bird/meshing/block_cyl_mesh_templates/coflowing/input.json  -t ../../../bird/meshing/block_cyl_mesh_templates/coflowing/topology.json -o system


# Mesh gen
blockMesh -dict system/blockMeshDict
cp -r IC/0 0

# Run
multiphaseEulerFoam



