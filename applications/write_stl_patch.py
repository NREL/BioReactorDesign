import numpy as np
import stl

from bird.preProcess.stl_patch.stl_bc import write_boundaries
from bird.preProcess.stl_patch.stl_shapes import *

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Spider Sparger STL")
    parser.add_argument(
        "-cr",
        "--centerRadius",
        type=float,
        metavar="",
        required=False,
        help="Radius of the center distributor",
        default=0.25,
    )
    parser.add_argument(
        "-na",
        "--nArms",
        type=int,
        metavar="",
        required=False,
        help="Number of spider arms",
        default=12,
    )
    parser.add_argument(
        "-aw",
        "--armsWidth",
        type=float,
        metavar="",
        required=False,
        help="Width of spider arms",
        default=0.1,
    )
    parser.add_argument(
        "-al",
        "--armsLength",
        type=float,
        metavar="",
        required=False,
        help="Length of spider arms",
        default=0.5,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="plot on screen"
    )
    args = parser.parse_args()

    bc_patch_dict = {}
    bc_patch_dict["spg"] = [
        {
            "type": "spider",
            "centerRad": args.centerRadius,
            "nArms": args.nArms,
            "widthArms": args.armsWidth,
            "lengthArms": args.armsLength,
            "centx": 0.0,
            "centy": 0.0,
            "centz": 0,
            "normal_dir": 1,
        }
    ]
    write_boundaries(bc_patch_dict)

    if args.verbose:
        # plot
        from bird.utilities.stl_plotting import plotSTL, plt, pretty_labels

        axes = plotSTL("spg.stl")
        pretty_labels("x", "y", zlabel="z", fontsize=14, ax=axes)
        plt.show()
