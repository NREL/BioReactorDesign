import argparse
import sys

import numpy as np

import os
import pickle

from bird.postProcess.conditional_mean import (
    compute_cond_mean,
    save_cond,
    sequencePlot,
)

parser = argparse.ArgumentParser(
    description="Compute conditional means of OpenFOAM fields"
)
parser.add_argument(
    "-f",
    "--caseFolder",
    type=str,
    metavar="",
    required=True,
    help="caseFolder to analyze",
    default=None,
)

parser.add_argument(
    "-vert",
    "--verticalDirection",
    type=int,
    metavar="",
    required=False,
    help="Index of vertical direction",
    default=1,
)
parser.add_argument(
    "-avg",
    "--windowAve",
    type=int,
    metavar="",
    required=False,
    help="Window Average",
    default=1,
)
parser.add_argument(
    "-fl",
    "--fields_list",
    nargs="+",
    help="List of fields to plot",
    default=[
        "CO.gas",
        "CO.liquid",
        "CO2.gas",
        "CO2.liquid",
        "H2.gas",
        "H2.liquid",
        "alpha.gas",
        "d.gas",
    ],
    required=False,
)
parser.add_argument(
        "-n",
        "--names",
        type=str,
        metavar="",
        required=False,
        help="names of cases",
        nargs="+",
        default=["test"],
)
parser.add_argument(
    "--diff_val_list",
    nargs="+",
    type=float,
    help="List of diffusivities",
    default=[],
    required=False,
)
parser.add_argument(
    "--diff_name_list",
    nargs="+",
    type=str,
    help="Diffusivities names",
    default=[],
    required=False,
)
args = parser.parse_args()


print(f"Case : {args.caseFolder}")

fields_cond = compute_cond_mean(
        args.caseFolder,
        args.verticalDirection,
        args.fields_list,
        args.windowAve,
        n_bins=32,
        diff_val_list=args.diff_val_list,
        diff_name_list=args.diff_name_list,
    )



with open(os.path.join(args.caseFolder, "cond.pkl"), "wb") as f:
    pickle.dump(fields_cond, f)
