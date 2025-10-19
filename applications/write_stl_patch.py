import os

import numpy as np
import stl

from bird import BIRD_PRE_PATCH_TEMP_DIR
from bird.preprocess.stl_patch.stl_bc import write_boundaries
from bird.preprocess.stl_patch.stl_shapes import *
from bird.utilities.parser import parse_json

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate boundary patch")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        metavar="",
        required=False,
        help="Boundary patch Json input",
        default=os.path.join(
            BIRD_PRE_PATCH_TEMP_DIR, "spider_spg/inlets_outlets.json"
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="plot on screen"
    )
    args = parser.parse_args()
    bc_patch_dict = parse_json(args.input)
    write_boundaries(bc_patch_dict)

    if args.verbose:
        # plot
        from bird.utilities.stl_plotting import plot_stl, plt, pretty_labels

        axes = plot_stl("inlets.stl")
        pretty_labels("x", "y", zlabel="z", fontsize=14, ax=axes)
        plt.show()
