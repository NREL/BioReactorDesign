import os
import tempfile
from pathlib import Path

import numpy as np
from prettyPlot.plotting import pretty_labels

from bird.preprocess.stl_patch.stl_bc import write_boundaries
from bird.utilities.parser import parse_json
from bird.utilities.stl_plotting import plot_stl


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

    input_dict = parse_json(
        os.path.join(BIRD_PRE_PATCH_TEMP_DIR, "spider_spg/inlets_outlets.json")
    )

    # Output to temporary directory and delete when done
    with tempfile.TemporaryDirectory() as tmpdirname:
        write_boundaries(input_dict, output_folder=tmpdirname)
        axes = plot_stl(os.path.join(tmpdirname, "inlets.stl"))
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
    input_dict = parse_json(
        os.path.join(
            BIRD_PRE_PATCH_TEMP_DIR, "loop_reactor_expl/inlets_outlets.json"
        )
    )

    # Output to temporary directory and delete when done
    with tempfile.TemporaryDirectory() as tmpdirname:
        write_boundaries(input_dict, output_folder=tmpdirname)
        axes = plot_stl(os.path.join(tmpdirname, "inlets.stl"))
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
    input_dict = parse_json(
        os.path.join(
            BIRD_PRE_PATCH_TEMP_DIR, "loop_reactor_branch/inlets_outlets.json"
        )
    )
    # Output to temporary directory and delete when done
    with tempfile.TemporaryDirectory() as tmpdirname:
        write_boundaries(input_dict, output_folder=tmpdirname)
        axes = plot_stl(os.path.join(tmpdirname, "inlets.stl"))
        pretty_labels("x", "y", zlabel="z", fontsize=14, ax=axes)

