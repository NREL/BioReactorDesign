#!/usr/bin/env bash
#
# Stitch a list of gmsh block meshes into ONE OpenFOAM mesh and run checkMesh.
#
# Each mesh is converted with gmshToFoam; mesh #0 becomes the master case that
# the rest are merged into; then each --stitch pair is coupled with integral
# (non -perfect) stitchMesh 
#
# Options:
#   --mesh   <file.msh>          (repeatable; order = stack order, #0 is master)
#   --stitch <master:slave>          (repeatable; integral, for non-matching faces)
#   --stitch-perfect <master:slave>  (repeatable; -perfect, for conformal faces)
#   --case   <dir>               (output case dir, default: stitched_case)

set -euo pipefail

CASE="stitched_case"
MESHES=()
STITCHES=()   # entries: "integral m:s" or "perfect m:s" (order preserved)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mesh)           MESHES+=("$2");             shift 2;;
    --stitch)         STITCHES+=("integral $2");  shift 2;;
    --stitch-perfect) STITCHES+=("perfect $2");   shift 2;;
    --case)           CASE="$2";                  shift 2;;
    *) echo "unknown argument: $1" >&2; exit 1;;
  esac
done

# >=1 mesh: with a single --mesh and no --stitch this just gmshToFoam+checkMesh
# one block (useful to isolate which block owns a checkMesh failure).
[[ ${#MESHES[@]} -ge 1 ]] || { echo "need at least one --mesh file" >&2; exit 1; }
command -v gmshToFoam >/dev/null 2>&1 || {
  echo "OpenFOAM not found on PATH — source OpenFOAM-9 first." >&2; exit 1; }

# minimal case skeleton (mesh utilities need controlDict/fvSchemes/fvSolution)
write_system() {
  local d="$1"; mkdir -p "$d/system" "$d/constant"
  cat > "$d/system/controlDict" <<'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application     checkMesh;
startFrom       startTime;  startTime 0;
stopAt          endTime;    endTime   1;
deltaT          1;          writeControl timeStep;  writeInterval 1;
EOF
  cat > "$d/system/fvSchemes" <<'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes        { default steadyState; }
gradSchemes       { default Gauss linear; }
divSchemes        { default none; }
laplacianSchemes  { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes     { default corrected; }
EOF
  cat > "$d/system/fvSolution" <<'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }
solvers {}
EOF
}

echo "==> master case: $CASE  (from ${MESHES[0]})"
rm -rf "$CASE"; write_system "$CASE"
gmshToFoam "${MESHES[0]}" -case "$CASE"

# convert + merge the remaining blocks into the master
for ((i=1; i<${#MESHES[@]}; i++)); do
  sub="${CASE}_add${i}"
  echo "==> add block $i: ${MESHES[$i]}"
  rm -rf "$sub"; write_system "$sub"
  gmshToFoam "${MESHES[$i]}" -case "$sub"
  # merge <master> <add> into master (OF-9 foundation syntax)
  mergeMeshes "$CASE" "$sub" -overwrite
  rm -rf "$sub"
done

# couple each interface with its chosen mode
for spec in "${STITCHES[@]}"; do
  mode="${spec%% *}"; pair="${spec#* }"
  master="${pair%%:*}"; slave="${pair##*:}"
  flags="-overwrite"; [[ "$mode" == perfect ]] && flags="$flags -perfect"
  echo "==> stitchMesh ($mode) $master $slave"
  stitchMesh $flags "$master" "$slave" -case "$CASE"
done

echo "==> checkMesh"
checkMesh -allGeometry -allTopology -case "$CASE"
echo "==> done. Mesh in $CASE/constant/polyMesh"
