import argparse
import os
import sys
from pathlib import Path

import numpy as np

from bird.meshing.stirred_tank_mesh import (
    get_reactor_geom,
    write_blocks,
    write_edges,
    write_ofoam_preamble,
    write_patches,
    write_vertices,
)


def test_stirred_tank():
    BIRD_STIRRED_TANK_MESH_TEMP_DIR = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "bird",
        "meshing",
        "stirred_tank_mesh_templates",
    )
    inp = os.path.join(
        BIRD_STIRRED_TANK_MESH_TEMP_DIR, "base_tank", "tank_par.yaml"
    )
    out = "tmp_blockMeshDict"
    with open(out, "w") as outfile:
        react = get_reactor_geom(inp)
        write_ofoam_preamble(outfile, react)
        write_vertices(outfile, react)
        write_edges(outfile, react)
        write_blocks(outfile, react)
        write_patches(outfile, react)
