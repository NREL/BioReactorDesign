#!/bin/bash
set -e  # Exit on any error
# Define what to do on error
trap 'echo "ERROR: Something failed! Running cleanup..."; ./Allclean' ERR

if ! type "blockMesh" &> /dev/null; then
    echo "<blockMesh> could not be found"
    echo "OpenFoam is likely not installed, skipping run"
else
    # Clean case
    ./Allclean
fi

if ! type "python" &> /dev/null; then
    echo "<python> could not be found"
    echo "Skipping Mesh generation"
else
    # Generate blockmeshDict
    python ../../applications/write_block_cyl_mesh.py -i ../../bird/meshing/block_cyl_mesh_templates/coflowing/input.json  -t ../../bird/meshing/block_cyl_mesh_templates/coflowing/topology.json -o system
fi


if ! type "blockMesh" &> /dev/null; then
    echo "<blockMesh> could not be found"
    echo "OpenFoam is likely not installed, skipping run"
else
    # Mesh gen
    blockMesh -dict system/blockMeshDict
    cp -r IC/0 0
    
    # Run
    birdmultiphaseEulerFoam
fi




