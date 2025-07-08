import os
from pathlib import Path

from prettyPlot.plotting import plt, pretty_labels

from bird.postprocess.conditional_mean import (
    compute_cond_mean,
    save_cond,
    sequencePlot,
)


def test_compute_cond():
    caseFolder = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "bird",
        "postprocess",
        "data_conditional_mean",
    )
    fields_list = [
        "CO.gas",
        "CO.liquid",
        "CO2.gas",
        "CO2.liquid",
        "H2.gas",
        "H2.liquid",
        "alpha.gas",
        "d.gas",
    ]
    fields_cond = compute_cond_mean(caseFolder, 1, fields_list, 2, n_bins=32)
    save_cond("cond.pkl", fields_cond)

    cond = {}
    cond[caseFolder] = fields_cond
    for field_name in fields_list:
        fig = plt.figure()
        plot_name = sequencePlot(cond, [caseFolder], field_name)
        pretty_labels(plot_name, "y [m]", 14)
        plt.close()
