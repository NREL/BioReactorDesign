ml anaconda3/2022.05
conda activate /projects/gas2fuels/conda_env/spargerDesign
python system/write_bmesh_file.py
conda deactivate
ml openfoam/11-craympich
foamCleanCase
rm -r 0
cp -r orig0 0
blockMesh -dict ./blockMeshDict_reactor
stitchMesh -perfect -overwrite inside_to_hub inside_to_hub_copy
stitchMesh -perfect -overwrite hub_to_rotor hub_to_rotor_copy
transformPoints "rotate=((0 0 1)(0 1 0))"
snappyHexMesh -overwrite
topoSet -dict system/topoSetDict_rm_inlet
createPatch -dict system/createPatchDict_inlet -overwrite
topoSet
createPatch -overwrite
module purge
source /projects/gas2fuels/load_OF9_pbe
setFields
rm -rf 0/meshPhi
touch sol.foam
