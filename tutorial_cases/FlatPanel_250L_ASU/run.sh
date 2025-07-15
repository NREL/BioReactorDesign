#!/bin/bash
set -e  # Exit on any error
# Define what to do on error
trap 'echo "ERROR: Something failed! Running cleanup..."; ./Allclean' ERR

bash presteps.sh
birdmultiphaseEulerFoam
