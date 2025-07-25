import os
from pathlib import Path

import numpy as np
from prettyPlot.plotting import pretty_labels

from bird.meshing._mesh_tools import parseJsonFile
from bird.preprocess.stl_patch.stl_bc import write_boundaries
from bird.utilities.stl_plotting import plotSTL


def test_spider_sparger():
    BIRD_PRE_PATCH_TEMP_DIR = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "bird",
        "preprocess",
        "stl_patch",
        "bc_patch_mesh_template",
    )

    input_dict = parseJsonFile(
        os.path.join(BIRD_PRE_PATCH_TEMP_DIR, "spider_spg/inlets_outlets.json")
    )
    write_boundaries(input_dict)
    # plot
    axes = plotSTL("inlets.stl")
    pretty_labels("x", "y", zlabel="z", fontsize=14, ax=axes)


def test_loop_reactor():
    BIRD_PRE_PATCH_TEMP_DIR = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "bird",
        "preprocess",
        "stl_patch",
        "bc_patch_mesh_template",
    )
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
    BIRD_PRE_PATCH_TEMP_DIR = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "bird",
        "preprocess",
        "stl_patch",
        "bc_patch_mesh_template",
    )
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
    from prettyPlot.plotting import plt

    test_spider_sparger()
    plt.show()
