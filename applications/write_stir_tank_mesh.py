import argparse
import os
import sys

import numpy as np

from brd import BRD_STIR_TANK_MESH_TEMP_DIR
from brd.meshing.stir_tank_mesh import (
    get_reactor_geom,
    write_blocks,
    write_edges,
    write_ofoam_preamble,
    write_patches,
    write_vertices,
)


def main():
    parser = argparse.ArgumentParser(description="Stir tank meshing")
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        metavar="",
        required=True,
        help="Output blockMeshDict",
        default="blockMeshDict",
    )
    parser.add_argument(
        "-i",
        "--react_in",
        type=str,
        metavar="",
        required=True,
        help="YAML file containing geometry details of reactor",
        default=os.path.join(
            BRD_STIR_TANK_MESH_TEMP_DIR, "base_tank", "tank_par.yaml"
        ),
    )
    args = parser.parse_args()
    with open(args.output_file, "w") as outfile:
        react = get_reactor_geom(args.react_in)
        write_ofoam_preamble(outfile, react)
        write_vertices(outfile, react)
        write_edges(outfile, react)
        write_blocks(outfile, react)
        write_patches(outfile, react)


if __name__ == "__main__":
    main()
