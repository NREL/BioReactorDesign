#module purge
#module load conda
rm ./blockMeshDict_reactor
rm -r dynamicCode
rm -r 0
cp -r 0.org 0
python3 system/write_bmesh_file.py
#module purge
#module load openmpi
#source /projects/nawihpro/OpenFoamV9/OpenFOAM-9/etc/bashrc 
blockMesh -dict ./blockMeshDict_reactor
stitchMesh -perfect -overwrite inside_to_hub inside_to_hub_copy
stitchMesh -perfect -overwrite hub_to_rotor hub_to_rotor_copy
transformPoints "rotate=((0 0 1)(0 1 0))"
surfaceToPatch -tol 1e-3 sparger.stl
export newmeshdir=$(foamListTimes -latestTime)
rm -rf constant/polyMesh/
cp -r $newmeshdir/polyMesh ./constant
rm -rf $newmeshdir
sed -i 's/patch0/inlet/g' ./constant/polyMesh/boundary
sed -i 's/zone0/inlet/g' ./constant/polyMesh/boundary
rm -rf 0
cp -r 0.org 0
setFields
