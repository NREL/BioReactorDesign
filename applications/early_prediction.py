from brd.postProcess.early_pred import multi_data_load, fit_and_ext, bayes_fit, plotAllEarly, plotAllEarly_uq
import argparse
from prettyPlot.plotting import plt

def main():
    from brd import BRD_EARLY_PRED_DATA_DIR
    parser = argparse.ArgumentParser(description="Early prediction")
    parser.add_argument(
        "-df",
        "--dataFolder",
        type=str,
        metavar="",
        required=True,
        help="Data folder containing multiple QOI time histories",
        default=BRD_EARLY_PRED_DATA_DIR,
    )
    parser.add_argument(
        "-func",
        "--functionalForm",
        type=str,
        metavar="",
        required=False,
        help="functional form used to perform extrapolation",
        default="doubleSigmoid",
    )
    args = parser.parse_args()

    if not args.functionalForm=="doubleSigmoid":
        raise NotImplementedError

    data_dict, color_files = multi_data_load(args.dataFolder)
    data_dict = fit_and_ext(data_dict)
    plotAllEarly(data_dict, color_files=color_files, chop=True, extrap=True)
    bayes_fit(data_dict)
    plotAllEarly_uq(data_dict, color_files=color_files)
    plt.show()

if __name__ == "__main__":
    main()
