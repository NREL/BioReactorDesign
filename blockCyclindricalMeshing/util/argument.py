import argparse


def initArg():
    # CLI
    parser = argparse.ArgumentParser(description="Block cylindrical meshing")
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        metavar="",
        required=False,
        help="Input file for meshing and geometry parameters",
        default="input",
    )
    parser.add_argument(
        "-g",
        "--geom_file",
        type=str,
        metavar="",
        required=False,
        help="Block description of the configuration",
        default="geometry",
    )
    parser.add_argument(
        "-o",
        "--out_folder",
        type=str,
        metavar="",
        required=False,
        help="Output folder for blockMeshDict",
        default="system",
    )
    args = parser.parse_args()

    return args
