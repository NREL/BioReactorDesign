import argparse


def initArgs():
    # CLI
    parser = argparse.ArgumentParser(
        description="Generate inhomogeneous boundary"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="plot on screen"
    )
    parser.add_argument(
        "-rc",
        "--r_const",
        type=float,
        metavar="",
        required=False,
        help="Constant radius value",
        default=0.1,
    )
    parser.add_argument(
        "-re",
        "--r_end",
        type=float,
        metavar="",
        required=False,
        help="End radius value",
        default=1,
    )
    parser.add_argument(
        "-pi",
        "--pore_in",
        type=float,
        metavar="",
        required=False,
        help="Pore diameter at center",
        default=3e-5,
    )
    parser.add_argument(
        "-po",
        "--pore_out",
        type=float,
        metavar="",
        required=False,
        help="Pore diameter at radius end",
        default=2e-5,
    )
    parser.add_argument(
        "-xc",
        "--xcent",
        type=float,
        metavar="",
        required=False,
        help="Column center x",
        default=0.0,
    )
    parser.add_argument(
        "-zc",
        "--zcent",
        type=float,
        metavar="",
        required=False,
        help="Column center z",
        default=0.0,
    )
    parser.add_argument(
        "-ds",
        "--diam_sparger",
        type=float,
        metavar="",
        required=False,
        help="Sparger diameter",
        default=0.15,
    )
    parser.add_argument(
        "-ugs",
        "--superf_vel",
        type=float,
        metavar="",
        required=False,
        help="Superficial velocity",
        default=0.01,
    )

    args = parser.parse_args()

    return args
