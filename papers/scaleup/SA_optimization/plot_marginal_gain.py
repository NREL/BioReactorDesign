import os

import pandas as pd
from prettyPlot.plotting import plt, pretty_labels

QOI_LABELS = {
    "qoi": {
        "label": "QOI",
        # "ylabel": r"Optimal QOI [kg$^2$/kWh$^2$]",
        "ylabel": r"Optimal QOI$_1$ [kg$^2$/kWh$^2$]",
    },
    "qoi_kla": {
        "label": "QOI Num",
        # "ylabel": r"Optimal QOI Num [kg$^2$/s$^2$]",
        "ylabel": r"Optimal QOI$_2$ [kg$^2$/s$^2$]",
    },
    "qoi_sum": {
        "label": "QOI Sum",
        # "ylabel": r"Optimal QOI Sum [kg/kWh]",
        "ylabel": r"Optimal QOI$_3$ [kg/kWh]",
    },
    "qoi_sum_kla": {
        "label": "QOI Sum Num",
        # "ylabel": r"Optimal QOI Sum Num [kg/s]",
        "ylabel": r"Optimal QOI$_4$ [kg/s]",
    },
}

LEVEL_TITLES = {
    "lev1": "3.6 L reactor",
    "lev6": r"608 m$^3$ reactor",
}


if __name__ == "__main__":
    df = pd.read_csv("marginal_gain_results.csv")

    for level in df["level"].unique():
        df_lev = df[df["level"] == level]
        for qoi_name in df_lev["qoi_name"].unique():
            df_q = df_lev[df_lev["qoi_name"] == qoi_name].sort_values(
                "n_spargers"
            )
            info = QOI_LABELS[qoi_name]

            n_spargers = df_q["n_spargers"].values[:-1]
            mean_optimum = df_q["optimal_qoi_mean"].values[:-1]
            ci95 = df_q["optimal_qoi_ci95"].values[:-1]

            fig, ax = plt.subplots()
            ax.plot(n_spargers, mean_optimum, linewidth=3, color="k")
            ax.fill_between(
                n_spargers,
                mean_optimum - ci95,
                mean_optimum + ci95,
                color="k",
                alpha=0.25,
                label="95% CI",
            )
            pretty_labels(
                r"N$_{\rm sparg}$",
                info["ylabel"],
                20,
                fontname="Times",
                grid=False,
                title=LEVEL_TITLES.get(level, level),
            )
            fname = f"marginal_gain_{level}_{qoi_name}"
            plt.savefig(f"{fname}.png", dpi=300)
            plt.savefig(f"{fname}.pdf")
            plt.close()
            print(f"Saved {fname}.png/.pdf")
