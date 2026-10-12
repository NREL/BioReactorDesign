#!/bin/bash
set -Eeuo pipefail   # -E: trap also fires inside functions/subshells
trap 'exit_code=$?
      echo "ERROR: Something failed! Running cleanup..."
      ./Allclean || true
      exit $exit_code' ERR

m4 system/conc_cylinder_mesh.m4 > system/blockMeshDict
rm -rf 0
cp -r 0.org 0
blockMesh
setFields
#decomposePar
