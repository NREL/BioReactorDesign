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
    BIRD_DIR=`python -c "import bird; print(bird.BIRD_DIR)"`
    # Generate blockmeshDict
    python ${BIRD_DIR}/../applications/write_block_cyl_mesh.py -i ${BIRD_DIR}/meshing/block_cyl_mesh_templates/sideSparger/input.json  -t ${BIRD_DIR}/meshing/block_cyl_mesh_templates/sideSparger/topology.json -o system
fi


if ! type "blockMesh" &> /dev/null; then
    echo "<blockMesh> could not be found"
    echo "OpenFoam is likely not installed, skipping run"
else
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
    birdmultiphaseEulerFoam
 
fi





