#!/bin/bash
set -e  # Exit on any error
# Define what to do on error
trap 'echo "ERROR: Something failed! Running cleanup..."; ./Allclean' ERR

cp -r 0.orig 0
m4 ./system/panel.m4 > ./system/blockMeshDict
blockMesh
setFields
