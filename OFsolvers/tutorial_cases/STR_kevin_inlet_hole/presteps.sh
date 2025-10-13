rm -r dynamicCode
rm -r 0
cp -r 0.org 0
python3 system/write_bmesh_file.py
blockMesh -dict ./blockMeshDict_reactor
stitchMesh -perfect -overwrite inside_to_hub inside_to_hub_copy
stitchMesh -perfect -overwrite hub_to_rotor hub_to_rotor_copy
transformPoints "rotate=((0 0 1)(0 1 0))"
snappyHexMesh -overwrite
topoSet
createPatch -overwrite
rm -r 0
cp -r 0.org 0
touch sol.foam
