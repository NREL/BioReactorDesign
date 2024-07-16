import os

import numpy as np
import stl

from bird import BIRD_PRE_DYNMIX_TEMP_DIR
from bird.meshing._mesh_tools import parseJsonFile
from bird.preprocess.dynamic_mixer.mixing_fvModels import *

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate dynamic mixer fvModels"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        metavar="",
        required=False,
        help="Mixers Json input",
        default=os.path.join(
            BIRD_PRE_DYNMIX_TEMP_DIR, "expl_list", "mixers.json"
        ),
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        metavar="",
        required=False,
        help="fvModels folder output",
        default=".",
    )
    args = parser.parse_args()
    dynMix_dict = parseJsonFile(args.input)
    write_fvModel(dynMix_dict, output_folder=args.output_folder)
