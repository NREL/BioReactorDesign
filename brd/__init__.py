"""Bio reactor design"""

import os

from brd.version import __version__

BRD_DIR = os.path.dirname(os.path.realpath(__file__))
BRD_MESH_DIR = os.path.join(BRD_DIR, "meshing")
BRD_POST_DIR = os.path.join(BRD_DIR, "postProcess")
BRD_BLOCK_CYL_MESH_TEMP_DIR = os.path.join(
    BRD_MESH_DIR, "block_cyl_mesh_templates"
)
BRD_BLOCK_CYL_CASE_TEMP_DIR = os.path.join(
    BRD_MESH_DIR, "block_cyl_case_templates"
)
BRD_STIR_TANK_MESH_TEMP_DIR = os.path.join(
    BRD_MESH_DIR, "stir_tank_mesh_templates"
)
BRD_STIR_TANK_CASE_TEMP_DIR = os.path.join(
    BRD_MESH_DIR, "stir_tank_case_templates"
)
BRD_EARLY_PRED_DATA_DIR = os.path.join(BRD_POST_DIR, "data_early")
BRD_INV_DIR = os.path.join(BRD_DIR, "inverse_modeling")
