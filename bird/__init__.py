"""Bio reactor design"""

import os

from bird.version import __version__

BIRD_DIR = os.path.dirname(os.path.realpath(__file__))
BIRD_MESH_DIR = os.path.join(BIRD_DIR, "meshing")
BIRD_POST_DIR = os.path.join(BIRD_DIR, "postProcess")
BIRD_BLOCK_CYL_MESH_TEMP_DIR = os.path.join(
    BIRD_MESH_DIR, "block_cyl_mesh_templates"
)
BIRD_BLOCK_CYL_CASE_TEMP_DIR = os.path.join(
    BIRD_MESH_DIR, "block_cyl_case_templates"
)
BIRD_STIR_TANK_MESH_TEMP_DIR = os.path.join(
    BIRD_MESH_DIR, "stir_tank_mesh_templates"
)
BIRD_STIR_TANK_CASE_TEMP_DIR = os.path.join(
    BIRD_MESH_DIR, "stir_tank_case_templates"
)
BIRD_EARLY_PRED_DATA_DIR = os.path.join(BIRD_POST_DIR, "data_early")
# BIRD_COND_MEAN_DATA_DIR = os.path.join(BIRD_POST_DIR, "data_conditional_mean")
BIRD_INV_DIR = os.path.join(BIRD_DIR, "inverse_modeling")
