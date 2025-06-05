# Clean case
module load conda
conda activate /projects/gas2fuels/conda_env/bird_wf
source /projects/gas2fuels/ofoam_cray_mpich/OpenFOAM-dev/etc/bashrc
./Allclean

BIRD_HOME=`python -c "import bird; print(bird.BIRD_DIR)"`

echo PRESTEP 1
# Generate blockmeshDict
python ${BIRD_HOME}/../applications/write_block_rect_mesh.py -i system/mesh.json -o system

# Generate boundary stl
python ${BIRD_HOME}/../applications/write_stl_patch.py -i system/inlets_outlets.json

# Generate mixers
python ${BIRD_HOME}/../applications/write_dynMix_fvModels.py -fs -i system/mixers.json -o constant

echo PRESTEP 2
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

conda deactivate

if [ -f qoi.txt ]; then
    rm qoi.txt
fi
if [ -f data/local/conv.npz ]; then
    rm data/local/conv.npz
fi

