root=`pwd`
caseFolder=case

python writeBlockMesh.py -i flatDonut/input.json -g flatDonut/geometry.json -o $caseFolder/system
#python writeBlockMesh.py -i sideSparger/input.json -g sideSparger/geometry.json -o $caseFolder/system

cd $caseFolder
blockMesh
transformPoints "scale=(0.001 0.001 0.001)"
transformPoints "rotate=((0 0 1) (0 1 0))"
cd $root
