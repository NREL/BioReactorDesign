#!/bin/sh
cd ${0%/*} || exit 1    # Run from this directory

# Source tutorial run functions
. $WM_PROJECT_DIR/bin/tools/RunFunctions

runApplication decomposePar -fileHandler collated
runParallel birdmultiphaseEulerFoam  -fileHandler collated
runApplication reconstructPar -newTimes
