root=`pwd`
caseFolder=case
#geometryFolder=flatDonut
#geometryFolder=sideSparger
#geometryFolder=baseColumn
#geometryFolder=baseColumn_refineSparg
geometryFolder=multiRing


python writeBlockMesh.py -i $geometryFolder/input.json -t $geometryFolder/topology.json -o $caseFolder/system

cd $caseFolder
blockMesh
transformPoints "scale=(0.001 0.001 0.001)"
transformPoints "rotate=((0 0 1) (0 1 0))"
cd $root
