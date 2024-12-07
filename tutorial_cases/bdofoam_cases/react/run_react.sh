cp -r ../nonreact/constant/polyMesh ./constant/polyMesh
cp -r ../nonreact/6000 .
decomposePar -fileHandler collated
srun -n 72 bdoFoam -parallel -fileHandler collated
