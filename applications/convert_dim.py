import argparse
import os
import shutil
import sys

from bird.preprocess.json_gen.generate_designs import (
    convert_case_dim,
    replace_str_in_file,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modify case dim")
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        metavar="",
        required=False,
        help="case folder input",
        default=None,
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        metavar="",
        required=False,
        help="case folder output",
        default=None,
    )
    parser.add_argument(
        "-f",
        "--dim_factor",
        type=float,
        metavar="",
        required=False,
        help="scaling factor",
        default=None,
    )

    args, unknown = parser.parse_known_args()

    convert_case_dim(args.input_folder, args.output_folder, args.dim_factor)
