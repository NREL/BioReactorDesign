import argparse
import os

from bird import BIRD_KLA_DATA_DIR
from bird.postprocess.kla_utils import compute_kla, print_res_dict

parser = argparse.ArgumentParser(description="KLA calculation with UQ")
parser.add_argument(
    "-i",
    "--data_file",
    type=str,
    metavar="",
    required=False,
    help="data_file",
    default=os.path.join(BIRD_KLA_DATA_DIR, "volume_avg.dat"),
)
parser.add_argument(
    "-ti",
    "--time_index",
    type=int,
    metavar="",
    required=False,
    help="column index for time",
    default=0,
)
parser.add_argument(
    "-ci",
    "--conc_index",
    type=int,
    metavar="",
    required=False,
    help="column index for concentration",
    default=1,
)
parser.add_argument(
    "-no_db",
    "--no_data_bootstrap",
    action="store_true",
    help="Do not do data bootstrapping",
)
parser.add_argument(
    "-mc",
    "--max_chop",
    type=int,
    metavar="",
    required=False,
    help="maximum number of early data to remove",
    default=None,
)

args, unknown = parser.parse_known_args()

if args.no_data_bootstrap:
    bootstrap = False
else:
    bootstrap = True

res_dict = compute_kla(
    filename=args.data_file,
    time_ind=args.time_index,
    conc_ind=args.conc_index,
    bootstrap=bootstrap,
    max_chop=args.max_chop,
)
print_res_dict(res_dict)
