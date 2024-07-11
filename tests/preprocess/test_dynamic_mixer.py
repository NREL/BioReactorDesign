import os

import numpy as np

from bird import BIRD_PRE_DYNMIX_TEMP_DIR
from bird.meshing._mesh_tools import parseJsonFile
from bird.preprocess.dynamic_mixer.mixing_fvModels import *


def test_expl_list():
    input_dict = parseJsonFile(
        os.path.join(BIRD_PRE_DYNMIX_TEMP_DIR, "expl_list/mixers.json")
    )
    write_fvModel(input_dict, output_folder=".")


# def test_loop_reactor():
#    input_dict = parseJsonFile(
#        os.path.join(
#            BIRD_PRE_PATCH_TEMP_DIR, "loop_reactor/inlets_outlets.json"
#        )
#    )
#    write_boundaries(input_dict)
#    # plot
#    axes = plotSTL("inlets.stl")
#    pretty_labels("x", "y", zlabel="z", fontsize=14, ax=axes)


if __name__ == "__main__":
    test_expl_list()
