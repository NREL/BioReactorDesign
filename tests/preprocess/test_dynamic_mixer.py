import os
from pathlib import Path

import numpy as np

from bird.preprocess.dynamic_mixer.mixing_fvModels import *
from bird.utilities.parser import parse_json


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
    input_dict = parse_json(
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
    input_dict = parse_json(
        os.path.join(
            BIRD_PRE_DYNMIX_TEMP_DIR, "loop_reactor_list", "mixers.json"
        )
    )
    write_fvModel(input_dict, output_folder=".")


if __name__ == "__main__":
    test_expl_list()
    test_loop_list()
