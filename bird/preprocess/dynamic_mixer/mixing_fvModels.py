import json
import sys

import numpy as np
import stl

from bird.meshing._mesh_tools import parseJsonFile
from bird.preprocess.dynamic_mixer.io_fvModels import *
from bird.preprocess.dynamic_mixer.mixer import Mixer


def check_input(input_dict):
    assert isinstance(input_dict, dict)


def write_fvModel(input_dict, output_folder="."):
    check_input(input_dict)
    write_preamble(output_folder)
    for mixer_dict in input_dict["mixers"]:
        mixer = Mixer()
        mixer.update_from_dict(mixer_dict)
        if mixer.ready:
            write_mixer(mixer, output_folder)
    write_end(output_folder)


if __name__ == "__main__":
    input_dict = parseJsonFile(
        os.path.join("mixing_template", "expl_list", "mixers.json"),
    )
    write_fvModel(input_dict)
