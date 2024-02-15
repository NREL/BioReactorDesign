import argparse
import os
import sys

import numpy as np

from bird import BIRD_BLOCK_CYL_MESH_TEMP_DIR
from bird.meshing.block_cyl_mesh import (
    assemble_geom,
    assemble_mesh,
    writeBlockMeshDict,
)


def main():
    parser = argparse.ArgumentParser(description="Block cylindrical meshing")
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        metavar="",
        required=True,
        help="Input file for meshing and geometry parameters",
        default="input.json",
    )
    parser.add_argument(
        "-t",
        "--topo_file",
        type=str,
        metavar="",
        required=True,
        help="Block description of the configuration",
        default="topology.json",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        metavar="",
        required=True,
        help="Output folder for blockMeshDict",
        default="system",
    )
    args = parser.parse_args()
    geomDict = assemble_geom(args.input_file, args.topo_file)
    meshDict = assemble_mesh(args.input_file, geomDict)
    writeBlockMeshDict(args.output_folder, geomDict, meshDict)


if __name__ == "__main__":
    main()
