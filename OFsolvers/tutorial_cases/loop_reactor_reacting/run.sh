if ! type "python" &> /dev/null; then
    echo "<python> could not be found"
    echo "Skipping Mesh generation"
else
    # Generate blockmeshDict
    python ../../../applications/write_block_rect_mesh.py -i system/mesh.json -o system
    
    # Generate boundary stl
    python ../../../applications/write_stl_patch.py -i system/inlets_outlets.json

    # Generate mixers
    #python ../../../applications/write_dynMix_fvModels.py -i system/mixers.json -o constant

fi


if ! type "blockMesh" &> /dev/null; then
    echo "<blockMesh> could not be found"
    echo "OpenFoam is likely not installed, skipping run"
else
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
    cp constant/polyMesh/boundary /tmp
    sed -i -e 's/inlets\.stl/inlet/g' /tmp/boundary
    cat /tmp/boundary > constant/polyMesh/boundary
    
    # Outlet BC
    surfaceToPatch -tol 1e-3 outlets.stl
    export newmeshdir=$(foamListTimes -latestTime)
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
    cp constant/phaseProperties_constantd constant/phaseProperties
    
    # Run
    biomultiphaseEulerFoam
fi




