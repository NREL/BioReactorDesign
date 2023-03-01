import argparse

def initArgs():
    # CLI
    parser = argparse.ArgumentParser(description="Generate Spider Sparger STL")
    parser.add_argument(
        "-cr",
        "--centerRadius",
        type=float,
        metavar="",
        required=False,
        help="Radius of the center distributor",
        default=0.25,
    )
    parser.add_argument(
        "-na",
        "--nArms",
        type=int,
        metavar="",
        required=False,
        help="Number of spider arms",
        default=12,
    )
    parser.add_argument(
        "-aw",
        "--armsWidth",
        type=float,
        metavar="",
        required=False,
        help="Width of spider arms",
        default=0.1,
    )
    parser.add_argument(
        "-al",
        "--armsLength",
        type=float,
        metavar="",
        required=False,
        help="Length of spider arms",
        default=0.5,
    )
    parser.add_argument(
        "-v", 
        "--verbose",
        action="store_true",
        help="plot on screen"
    )

    args = parser.parse_args()

    return args
