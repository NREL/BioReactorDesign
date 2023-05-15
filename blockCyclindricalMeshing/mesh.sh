root=`pwd`
caseFolder=case

#python writeBlockMesh.py -i flatDonut/input.json -t flatDonut/topology.json -o $caseFolder/system
#python writeBlockMesh.py -i sideSparger/input.json -t sideSparger/topology.json -o $caseFolder/system
#python writeBlockMesh.py -i baseColumn/input.json -t baseColumn/topology.json -o $caseFolder/system
python writeBlockMesh.py -i baseColumn_refineSparg/input.json -t baseColumn_refineSparg/topology.json -o $caseFolder/system

cd $caseFolder
blockMesh
transformPoints "scale=(0.001 0.001 0.001)"
transformPoints "rotate=((0 0 1) (0 1 0))"
cd $root
