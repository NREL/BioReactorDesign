# Clean case
#module load anaconda3/2023
#conda activate /projects/gas2fuels/conda_env/bird
#source /projects/gas2fuels/ofoam_cray_mpich/OpenFOAM-dev/etc/bashrc
./Allclean
. $WM_PROJECT_DIR/bin/tools/RunFunctions

BIRD_HOME=`python -c "import bird; print(bird.BIRD_DIR)"`


echo PRESTEP 1
# Generate blockmeshDict
python $BIRD_HOME/../applications/write_block_cyl_mesh.py -i system/mesh.json -t system/topology.json -o system

echo PRESTEP 2
# Mesh gen
blockMesh -dict system/blockMeshDict
transformPoints "rotate=((0 0 1) (0 1 0))"
transformPoints "scale=(0.001 0.001 0.001)"
# setup IC

# Scale
transformPoints "scale=(0.19145161188225573 0.19145161188225573 0.19145161188225573)"

# Inlet BC
topoSet
createPatch -overwrite
cp -r 0.orig 0
setFields
postProcess -func 'patchIntegrate(patch="inlet", field="one")'
rm 0/one
postProcess -func writeCellVolumes
writeMeshObj




echo PRESTEP 3
python writeGlobalVars.py

