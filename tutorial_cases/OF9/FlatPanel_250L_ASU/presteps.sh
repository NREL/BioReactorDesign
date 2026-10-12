#!/bin/bash
set -Eeuo pipefail   # -E: trap also fires inside functions/subshells
trap 'exit_code=$?
      echo "ERROR: Something failed! Running cleanup..."
      ./Allclean || true
      exit $exit_code' ERR

cp -r 0.orig 0
m4 ./system/panel.m4 > ./system/blockMeshDict
blockMesh
setFields
