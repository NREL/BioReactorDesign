"""Bio reactor design"""

import os

from bird.version import __version__

BIRD_DIR = os.path.dirname(os.path.realpath(__file__))
BIRD_MESH_DIR = os.path.join(BIRD_DIR, "meshing")
BIRD_POST_DIR = os.path.join(BIRD_DIR, "postprocess")
BIRD_PRE_DIR = os.path.join(BIRD_DIR, "preprocess")
BIRD_BLOCK_CYL_MESH_TEMP_DIR = os.path.join(
    BIRD_MESH_DIR, "block_cyl_mesh_templates"
)
BIRD_BLOCK_CYL_CASE_TEMP_DIR = os.path.join(
    BIRD_MESH_DIR, "block_cyl_case_templates"
)
BIRD_BLOCK_RECT_MESH_TEMP_DIR = os.path.join(
    BIRD_MESH_DIR, "block_rect_mesh_templates"
)
BIRD_BLOCK_RECT_CASE_TEMP_DIR = os.path.join(
    BIRD_MESH_DIR, "block_rect_case_templates"
)
BIRD_STIRRED_TANK_MESH_TEMP_DIR = os.path.join(
    BIRD_MESH_DIR, "stirred_tank_mesh_templates"
)
BIRD_STIRRED_TANK_CASE_TEMP_DIR = os.path.join(
    BIRD_MESH_DIR, "stirred_tank_case_templates"
)
BIRD_PRE_PATCH_TEMP_DIR = os.path.join(
    BIRD_PRE_DIR, "stl_patch", "bc_patch_mesh_template"
)
BIRD_PRE_DYNMIX_TEMP_DIR = os.path.join(
    BIRD_PRE_DIR, "dynamic_mixer", "mixing_template"
)
BIRD_EARLY_PRED_DATA_DIR = os.path.join(BIRD_POST_DIR, "data_early")
BIRD_KLA_DATA_DIR = os.path.join(BIRD_POST_DIR, "data_kla")
BIRD_INV_DIR = os.path.join(BIRD_DIR, "inverse_modeling")
