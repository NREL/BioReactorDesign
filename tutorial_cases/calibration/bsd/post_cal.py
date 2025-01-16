import corner
import jax.numpy as jnp
from prettyPlot.plotting import *


def post_process_cal(
    labels_np,
    labels,
    np_mcmc_samples,
    rangex,
    forward_range,
    data_x,
    data_y,
    sigma,
    args,
):
    # Post process
    ranges = []
    ranges.append((0.7, 1.2))
    ranges.append((0.01, 2.2))
    if len(labels_np) == 3:
        ranges.append((0.001, 0.1))
    if (
        args.true_breakup_rate_factor is not None
        and args.true_coalescence_rate is not None
    ):
        if len(labels_np) == 3:
            truths = [
                args.true_breakup_rate_factor,
                args.true_coalescence_rate,
                0,
            ]
        else:
            truths = [
                args.true_breakup_rate_factor,
                args.true_coalescence_rate,
            ]
    else:
        truths = None

    labels_np_disp = labels_np.copy()
    ind_beff_fact = labels_np_disp.index("beff_fact")
    labels_np_disp[ind_beff_fact] = r"$B_{\rm f}$"
    ind_ceff = labels_np_disp.index("ceff")
    labels_np_disp[ind_ceff] = r"$C_{\rm r}$"
    fig = corner.corner(
        np_mcmc_samples,
        truths=truths,
        labels=labels_np_disp,
        truth_color="r",
        bins=50,
        range=ranges,
    )
    filename = ""
    filename += "Surr"
    if args.cal_err:
        filename += "_cal"
    else:
        filename += "_opt"
    if args.ternary:
        filename += "_ternary"
    else:
        filename += "_binary"
    filename += f"_corner"

    for ax in fig.get_axes():
        ax.tick_params(
            axis="both", labelsize=16
        )  # Customize font size, line width, and tick length
        ax.xaxis.label.set_fontweight("bold")  # Set the X-axis label to bold
        ax.yaxis.label.set_fontweight("bold")  # Set the Y-axis label to bold
        ax.xaxis.label.set_font(
            "serif"
        )  # Set the X-axis label to bold
        ax.yaxis.label.set_font(
            "serif"
        )  # Set the Y-axis label to bold
        ax.xaxis.label.set_fontsize(16)  # Set the X-axis label to bold
        ax.yaxis.label.set_fontsize(16)  # Set the Y-axis label to bold
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontname("Times New Roman")
            tick.set_fontweight("bold")
    for ax in fig.get_axes():
        ax.set_xlabel(ax.get_xlabel(), fontweight="bold")
        ax.set_ylabel(ax.get_ylabel(), fontweight="bold")
    plt.savefig(filename + ".png")
    plt.savefig(filename + ".eps")

    # Convergence
    fig, axes = plt.subplots(2, sharex=True)
    for i in range(2):
        ax = axes[i]
        ax.plot(np_mcmc_samples[:, i], "k", alpha=0.3, rasterized=True)
        ax.set_ylabel(labels[i])

    nsamples = np_mcmc_samples.shape[0]
    print("Num samples = ", nsamples)
    realization = []
    for i in range(nsamples):
        y = forward_range(np_mcmc_samples[i, :2].astype("float32"))
        realization.append(y)
    realization = np.array(realization)

    mean_real = np.mean(realization, axis=0)
    min_real = np.min(realization, axis=0)
    max_real = np.max(realization, axis=0)
    std97_5_real = np.percentile(realization, 97.5, axis=0)
    std2_5_real = np.percentile(realization, 2.5, axis=0)

    fig = plt.figure()
    plt.plot(data_x, data_y, "o", color="r", markersize=7, label="Data")
    plt.plot(data_x, data_y - 2 * sigma, "--", color="r")
    plt.plot(
        data_x,
        data_y + 2 * sigma,
        "--",
        color="r",
        label="95% Exp. confidence interval",
    )

    plt.plot(
        rangex, mean_real, color="k", linewidth=3, label="mean degradation"
    )
    plt.plot(
        rangex,
        std97_5_real,
        "--",
        color="k",
        linewidth=3,
        label="95% Model confidence interval",
    )
    plt.plot(rangex, std2_5_real, "--", color="k", linewidth=3)
    pretty_labels(
        "Bubble diameter [m]", "PDF", 16, title=f"Noise + missing phys. unc. = {sigma:.2g}"
    )
    filename = ""
    filename += "Surr"
    if args.cal_err:
        filename += "_cal"
    else:
        filename += "_opt"
    if args.ternary:
        filename += "_ternary"
    else:
        filename += "_binary"
    filename += f"_prop"
    plt.savefig(filename + ".png")
    plt.savefig(filename + ".eps")
    #plt.show()
