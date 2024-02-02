import argparse

from prettyPlot.plotting import plt, pretty_labels

# from brd import BRD_COND_MEAN_DATA_DIR
from brd.postProcess.conditional_mean import (
    compute_cond_mean,
    save_cond,
    sequencePlot,
)


def main():

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
        default="brd/postProcess/data_conditional_mean",
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
        default=["CO2.gas", "alpha.gas"],
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

    fields_cond = compute_cond_mean(
        args.caseFolder,
        args.verticalDirection,
        args.fields_list,
        args.windowAve,
        n_bins=32,
        diff_val_list=args.diff_val_list,
        diff_name_list=args.diff_name_list,
    )

    cond = {}
    cond[args.caseFolder] = fields_cond
    for field_name in args.fields_list:
        fig = plt.figure()
        plot_name = sequencePlot(cond, [args.caseFolder], field_name)
        pretty_labels(plot_name, "y [m]", 14)

    plt.show()


if __name__ == "__main__":
    main()
