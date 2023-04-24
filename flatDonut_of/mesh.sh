cd system 
python writeBlockMesh.py input
cd ..
blockMesh
transformPoints "scale=(0.001 0.001 0.001)"
transformPoints "rotate=((0 0 1) (0 1 0))" 
