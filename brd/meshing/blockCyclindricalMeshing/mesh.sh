root=`pwd`
caseFolder=case
geometryFolder=flatDonut
#geometryFolder=sideSparger
#geometryFolder=baseColumn
#geometryFolder=baseColumn_projected
#geometryFolder=baseColumn_refineSparg
#geometryFolder=multiRing_rad_width
#geometryFolder=template_flatDonut
#geometryFolder=template_sideSparger
#geometryFolder=testMR
#yygeometryFolder=multiRing_coarse
#geometryFolder=flatDonut_slot
#geometryFolder=flatDonut_projected
#geometryFolder=multiRing_simple_projected
#geometryFolder=circle
#geometryFolder=circle_projected



python writeBlockMesh.py -i $geometryFolder/input.json -t $geometryFolder/topology.json -o $caseFolder/system

cd $caseFolder
blockMesh
transformPoints "scale=(0.001 0.001 0.001)"
transformPoints "rotate=((0 0 1) (0 1 0))"
cd $root
