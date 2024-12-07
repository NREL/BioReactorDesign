#!/bin/bash
rm -rf 0
cp -r 0.org 0
blockMesh
setFields
decomposePar -fileHandler collated
srun -n 72 bdoFoam -parallel -fileHandler collated
