"""Bio reactor design"""
import os
from brd.version import __version__

BRD_DIR = os.path.dirname(os.path.realpath(__file__))
BRD_MESH_DIR = os.path.join(BRD_DIR, "meshing")
BRD_BLOCK_CYL_MESH_TEMP_DIR = os.path.join(BRD_MESH_DIR, "block_cyl_mesh_templates")
BRD_BLOCK_CYL_CASE_TEMP_DIR = os.path.join(BRD_MESH_DIR, "block_cyl_case_templates")
BRD_INV_DIR = os.path.join(BRD_DIR, "inverse_modeling")
