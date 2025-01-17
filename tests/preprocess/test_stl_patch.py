import os

import numpy as np

from bird import BIRD_PRE_PATCH_TEMP_DIR
from bird.meshing._mesh_tools import parseJsonFile
from bird.preprocess.stl_patch.stl_bc import write_boundaries
from bird.utilities.stl_plotting import plotSTL, pretty_labels


def test_spider_sparger():
    input_dict = parseJsonFile(
        os.path.join(BIRD_PRE_PATCH_TEMP_DIR, "spider_spg/inlets_outlets.json")
    )
    write_boundaries(input_dict)
    # plot
    axes = plotSTL("inlets.stl")
    pretty_labels("x", "y", zlabel="z", fontsize=14, ax=axes)


def test_loop_reactor():
    input_dict = parseJsonFile(
        os.path.join(
            BIRD_PRE_PATCH_TEMP_DIR, "loop_reactor_expl/inlets_outlets.json"
        )
    )
    write_boundaries(input_dict)
    # plot
    axes = plotSTL("inlets.stl")
    pretty_labels("x", "y", zlabel="z", fontsize=14, ax=axes)


def test_loop_reactor_branch():
    input_dict = parseJsonFile(
        os.path.join(
            BIRD_PRE_PATCH_TEMP_DIR, "loop_reactor_branch/inlets_outlets.json"
        )
    )
    write_boundaries(input_dict)
    # plot
    axes = plotSTL("inlets.stl")
    pretty_labels("x", "y", zlabel="z", fontsize=14, ax=axes)


if __name__ == "__main__":
    test_spider_sparger()
