if ! type "python" &> /dev/null; then
    echo "<python> could not be found"
    echo "Skipping Mesh generation"
else
    # Generate blockmeshDict
    python system/write_bmesh_file.py 
fi


if ! type "blockMesh" &> /dev/null; then
    echo "<blockMesh> could not be found"
    echo "OpenFoam is likely not installed, skipping run"
else
    # Clean case
    foamCleanCase
    
    rm -r 0
    cp -r orig0 0

    # Mesh gen
    blockMesh -dict ./blockMeshDict_reactor
    stitchMesh -perfect -overwrite inside_to_hub inside_to_hub_copy
    stitchMesh -perfect -overwrite hub_to_rotor hub_to_rotor_copy
    transformPoints "rotate=((0 0 1)(0 1 0))"
    snappyHexMesh -overwrite
    topoSet -dict system/topoSetDict_rm_inlet
    createPatch -dict system/createPatchDict_inlet -overwrite
    topoSet
    createPatch -overwrite
    setFields
    rm -rf 0/meshPhi
    touch sol.foam
    
    # Run
    multiphaseEulerFoam
fi




