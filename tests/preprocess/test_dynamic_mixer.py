import os
from pathlib import Path

import numpy as np

from bird.meshing._mesh_tools import parseJsonFile
from bird.preprocess.dynamic_mixer.mixing_fvModels import *


def test_expl_list():
    BIRD_PRE_DYNMIX_TEMP_DIR = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "bird",
        "preprocess",
        "dynamic_mixer",
        "mixing_template",
    )
    input_dict = parseJsonFile(
        os.path.join(BIRD_PRE_DYNMIX_TEMP_DIR, "expl_list", "mixers.json")
    )
    write_fvModel(input_dict, output_folder=".")


def test_loop_list():
    BIRD_PRE_DYNMIX_TEMP_DIR = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "bird",
        "preprocess",
        "dynamic_mixer",
        "mixing_template",
    )
    input_dict = parseJsonFile(
        os.path.join(
            BIRD_PRE_DYNMIX_TEMP_DIR, "loop_reactor_list", "mixers.json"
        )
    )
    write_fvModel(input_dict, output_folder=".")


if __name__ == "__main__":
    test_expl_list()
    test_loop_list()
