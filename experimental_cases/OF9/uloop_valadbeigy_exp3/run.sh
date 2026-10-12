#!/bin/bash
### OpenFOAM command
./Allclean
set -e  # Exit on any error
trap 'echo "ERROR: Something failed! Running cleanup..."; ./Allclean' ERR

### BiRD command
echo PRESTEP 1
BIRD_DIR=$(python -c "import bird; print(bird.BIRD_DIR)")
APPLICATIONS=$(dirname "$BIRD_DIR")/applications

python "$APPLICATIONS/write_stl_patch.py" -i system/inlets_outlets.json
python "$APPLICATIONS/write_dynMix_fvModels.py" -i system/mixers.json -o constant

echo PRESTEP 2
python build_uloop_hex.py
bash stitch_and_check.sh \
  --mesh blockC.msh \
  --mesh blockB.msh --mesh blockA.msh \
  --stitch int_B_top:int_C_bot \
  --stitch int_A_legL:int_B_legL \
  --stitch int_A_legR:int_B_legR \
  --case stitched_case_uloop
touch stitched_case_uloop/test.foam
cp -r stitched_case_uloop/constant/polyMesh constant/polyMesh
createPatch -overwrite
transformPoints "rotate=((0 0 1) (0 1 0))"

# Make a local tmp folder to preprocess the mesh
mkdir tmp

# --- sparger ---
surfaceToPatch -tol 1e-3 sparger.stl
export newmeshdir=$(foamListTimes -latestTime)
rm -rf constant/polyMesh/
cp -r $newmeshdir/polyMesh ./constant
rm -rf $newmeshdir
cp constant/polyMesh/boundary tmp
sed -i -e 's/sparger\.stl/sparger/g' tmp/boundary
cat tmp/boundary > constant/polyMesh/boundary

# --- dye_inlet ---
surfaceToPatch -tol 1e-3 dye_inlet.stl
export newmeshdir=$(foamListTimes -latestTime)
rm -rf constant/polyMesh/
cp -r $newmeshdir/polyMesh ./constant
rm -rf $newmeshdir
cp constant/polyMesh/boundary tmp
sed -i -e 's/dye_inlet\.stl/dye_inlet/g' tmp/boundary
cat tmp/boundary > constant/polyMesh/boundary

foamDictionary constant/polyMesh/boundary -entry entry0/walls/type -set wall
foamDictionary constant/polyMesh/boundary -entry entry0/dye_inlet/type -set wall

cp -r 0.orig 0

DYE_START=$(grep -E '^[[:space:]]*dyeStart[[:space:]]' constant/globalVars_temp | head -1 | sed -E 's/^[[:space:]]*dyeStart[[:space:]]+([0-9.eE+-]+).*/\1/')
DYE_STOP=$(grep -E '^[[:space:]]*dyeStop[[:space:]]' constant/globalVars_temp | head -1 | sed -E 's/^[[:space:]]*dyeStop[[:space:]]+([0-9.eE+-]+).*/\1/')
echo "Dye injection window: dyeStart=$DYE_START dyeStop=$DYE_STOP"
grep -rl '__DYE_START__\|__DYE_STOP__' 0 | xargs -r sed -i "s/__DYE_START__/${DYE_START}/g; s/__DYE_STOP__/${DYE_STOP}/g"

setFields

postProcess -func 'patchIntegrate(patch="sparger", field="alpha.gas")'
postProcess -func 'patchIntegrate(patch="dye_inlet", field="alpha.liquid")'
postProcess -func 'patchIntegrate(patch="dye_inlet", field="alpha.gas")'
postProcess -func writeCellVolumes
writeMeshObj

echo PRESTEP 3
python writeGlobalVars.py
cp constant/phaseProperties_constantd constant/phaseProperties


birdmultiphaseEulerFoam
