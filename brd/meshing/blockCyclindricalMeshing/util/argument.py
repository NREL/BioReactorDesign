import argparse


def initArg():
    # CLI
    parser = argparse.ArgumentParser(description="Block cylindrical meshing")
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        metavar="",
        required=True,
        help="Input file for meshing and geometry parameters",
        default="input.json",
    )
    parser.add_argument(
        "-t",
        "--topo_file",
        type=str,
        metavar="",
        required=True,
        help="Block description of the configuration",
        default="topology.json",
    )
    parser.add_argument(
        "-o",
        "--out_folder",
        type=str,
        metavar="",
        required=True,
        help="Output folder for blockMeshDict",
        default="system",
    )
    args = parser.parse_args()

    return args
