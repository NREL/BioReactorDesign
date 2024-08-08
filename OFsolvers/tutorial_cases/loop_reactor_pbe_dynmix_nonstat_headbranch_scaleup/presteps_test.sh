python ../../../applications/write_block_rect_mesh.py -i system/mesh.json -o system

# Generate boundary stl
python ../../../applications/write_stl_patch.py -i system/inlets_outlets.json

# Generate mixers
python ../../../applications/write_dynMix_fvModels.py -i system/mixers.json -o constant

