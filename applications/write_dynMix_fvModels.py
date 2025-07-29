import os

import numpy as np
import stl

from bird import BIRD_PRE_DYNMIX_TEMP_DIR
from bird.preprocess.dynamic_mixer.mixing_fvModels import *
from bird.utilities.parser import parse_json

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
    parser.add_argument(
        "-fs",
        "--force_sign",
        action="store_true",
        help="Force mixing source sign",
    )
    args = parser.parse_args()
    dynMix_dict = parse_json(args.input)
    write_fvModel(
        dynMix_dict,
        output_folder=args.output_folder,
        force_sign=args.force_sign,
    )
