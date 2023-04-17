cd system 
python writeBlockMesh.py input
cd ..
blockMesh
transformPoints "scale=(0.001 0.001 0.001)"
 
