import argparse
import sys

import numpy as np

from bird.meshing.stl_shapes import make_spider, from_dict_to_stl
from bird.meshing.stl_mesh_tools import saveSTL

if __name__ == "__main__":
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
    # Spider
    spider_mesh, spider_area = make_spider(
        centerRad=args.centerRadius,
        nArms=args.nArms,
        widthArms=args.armsWidth,
        lengthArms=args.armsLength,
        center=(0,0,0),
        normal_dir=1
    )
    print(f"\tglobalArea = {spider_area}")
    spider_stl = from_dict_to_stl(spider_mesh)
    saveSTL(spider_stl, filename="spg.stl")

    if args.verbose:
        # plot
        from bird.utilities.stl_plotting import plotSTL, plt, pretty_labels

        axes = plotSTL("spg.stl")
        pretty_labels("x", "y", zlabel="z", fontsize=14, ax=axes)
        plt.show()
