import os

import corner
import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from prettyPlot.plotting import *
from scipy.optimize import curve_fit


def plotAllEarly(data_dict, color_files=None, chop=False, extrap=False):
    fig = plt.figure()
    for idat, datf in enumerate(data_dict):
        if color_files is not None:
            color = color_files[idat]
        else:
            color = None
        if extrap:
            plt.plot(
                data_dict[datf]["textrap"],
                100 * data_dict[datf]["yextrap"],
                "--",
                color=color,
                label=f"{datf} y={100 * data_dict[datf]['yextrap'][-1]:.2f}",
            )
        if chop:
            lim = data_dict[datf]["lim"]
            plt.plot(
                data_dict[datf]["t"][lim:],
                100 * data_dict[datf]["y"][lim:],
                "o",
                color=color,
            )
        else:
            plt.plot(
                data_dict[datf]["t"], 100 * data_dict[datf]["y"], color=color
            )
    pretty_labels("time [s]", "yield [%]", 14)
    if extrap:
        pretty_legend(fontsize=14)


def plotAllEarly_uq(data_dict, color_files=None):
    fig = plt.figure()
    for idat, datf in enumerate(data_dict):
        color = color_files[idat]
        text = data_dict[datf]["textrap"]
        t = data_dict[datf]["t"]
        y = data_dict[datf]["y"] * 100
        med_real = data_dict[datf]["med_real"] * 100
        mean_real = data_dict[datf]["mean_real"] * 100
        std16_real = data_dict[datf]["std16_real"] * 100
        std84_real = data_dict[datf]["std84_real"] * 100

        plt.plot(text, med_real, color=color, linewidth=3)
        plt.fill_between(text, std16_real, std84_real, color=color, alpha=0.3)
        plt.plot(
            t,
            y,
            "o",
            color=color,
            markersize=5,
            linewidth=3,
            label=f"{datf} y={med_real[-1]:.2f}+/-{(std84_real[-1]-std16_real[-1])/2:.2f}",
        )
        pretty_labels("time [s]", "yield [%]", 14)
        pretty_legend(fontsize=14)


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth


def multi_data_load(data_root, tmax=600, data_files=None, color_files=None):
    if data_files is None:
        data_files = os.listdir(data_root)
    if color_files is None:
        color_files = [
            str(i * 0.75 / (len(data_files))) for i in range(len(data_files))
        ]
    data_dict = {}
    for idat, datf in enumerate(data_files):
        filename = os.path.join(data_root, datf)
        A = np.loadtxt(filename)
        data_dict[datf] = {}
        data_dict[datf]["t"] = A[:, 0]
        data_dict[datf]["y"] = A[:, 5] / (A[:, 4] * 16 / 44 + A[:, 5])
        # chop data before increase and right after t=10s
        increase_ind_arr = np.argwhere(np.diff(data_dict[datf]["y"]) > 0)
        increase_ind = increase_ind_arr[
            np.argwhere(data_dict[datf]["t"][increase_ind_arr] > 10)[0][0]
        ][0]
        print(
            f"data {datf} first time {data_dict[datf]['t'][increase_ind]:.2f}"
        )
        data_dict[datf]["lim"] = increase_ind
        y_fit = (
            data_dict[datf]["y"][increase_ind:]
            - data_dict[datf]["y"][increase_ind]
        )
        t_fit = (
            data_dict[datf]["t"][increase_ind:]
            - data_dict[datf]["t"][increase_ind]
        )
        data_dict[datf]["t_fit"] = t_fit
        data_dict[datf]["y_fit"] = y_fit
        data_dict[datf]["textrap"] = np.linspace(0, tmax, min(tmax, 1000))

    return data_dict, color_files


def sigm_fit(x, a, b, c, d, e, f):
    return 2 * c * (
        0.5 - 1 / (1 + np.exp(np.clip((a * x) ** b, a_min=None, a_max=20)))
    ) + 2 * f * (
        0.5 - 1 / (1 + np.exp(np.clip((d * x) ** e, a_min=None, a_max=20)))
    )


def fit_and_ext(
    data_dict,
    func=sigm_fit,
    bounds=([0, 0, 0, 0, 0, 0], [0.05, 1.7, np.inf, 0.015, 1.3, np.inf]),
):
    for idat, datf in enumerate(data_dict):
        popt, pcov = curve_fit(
            func,
            data_dict[datf]["t_fit"],
            data_dict[datf]["y_fit"],
            bounds=bounds,
        )
        data_dict[datf]["yextrap"] = func(data_dict[datf]["textrap"], *popt)
        print(f"data {datf} coeff {popt}")
        lim_ind = data_dict[datf]["lim"]
        data_dict[datf]["textrap"] += data_dict[datf]["t"][lim_ind]
        data_dict[datf]["yextrap"] += data_dict[datf]["y"][lim_ind]

    return data_dict


def sigm_fit_jax(theta, x):
    a, b, M, c, d, N, sigma = theta
    F = (
        2
        * M
        * (0.5 - 1 / (1 + jnp.exp(jnp.clip((a * x) ** b, min=None, max=20))))
    )
    F += (
        2
        * N
        * (0.5 - 1 / (1 + jnp.exp(jnp.clip((c * x) ** d, min=None, max=20))))
    )
    return F


def bayes_step(x, y=None, y_err=0.1):
    # define parameters (incl. prior ranges)
    a = numpyro.sample("a", dist.Uniform(0, 0.05))
    b = numpyro.sample("b", dist.Uniform(0, 1.7))
    M = numpyro.sample("M", dist.Uniform(0, 1))
    c = numpyro.sample("c", dist.Uniform(0, 0.015))
    d = numpyro.sample("d", dist.Uniform(0.5, 1.3))
    N = numpyro.sample("N", dist.Uniform(0, 1))
    sigma = numpyro.sample("sigma", dist.Uniform(1e-2, 10))

    # implement the model
    # needs jax numpy for differentiability here
    y_model = sigm_fit_jax([a, b, M, c, d, N, sigma], x)

    # notice that we clamp the outcome of this sampling to the observation y
    numpyro.sample("obs", dist.Normal(y_model, sigma), obs=y)


def bayes_fit(data_dict, num_warmup=1000, num_samples=500):
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    for idat, datf in enumerate(data_dict):
        # Hamilton Markov Chain (HMC) with no u turn sampling (NUTS)
        kernel = NUTS(bayes_step, target_accept_prob=0.9)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
        mcmc.run(
            rng_key_, x=data_dict[datf]["t_fit"], y=data_dict[datf]["y_fit"]
        )
        mcmc.print_summary()

        # Draw samples
        hmc_samples = mcmc.get_samples()
        labels = list(hmc_samples.keys())
        nsamples = len(hmc_samples[labels[0]])
        nparams = len(labels)
        np_hmc_samples = np.zeros((nsamples, nparams))
        for ilabel, label in enumerate(labels):
            if label == "a":
                nplabel = 0
            if label == "b":
                nplabel = 1
            if label == "M":
                nplabel = 2
            if label == "c":
                nplabel = 3
            if label == "d":
                nplabel = 4
            if label == "N":
                nplabel = 5
            if label == "sigma":
                nplabel = 6
            np_hmc_samples[:, nplabel] = np.array(hmc_samples[label])

        ## Post process
        # fig = corner.corner(np_hmc_samples, truths=theta, labels=labels)
        #
        #
        ## Convergence
        # fig, axes = plt.subplots(nparams, sharex=True)
        # for i in range(nparams):
        #    ax = axes[i]
        #    ax.plot(np_hmc_samples[:, i], "k", alpha=0.3, rasterized=True)
        #    ax.set_ylabel(labels[i])

        # data_dict[datf]["yextrap"] = f(data_dict[datf]["textrap"], *popt)
        # data_dict[datf]["textrap"] += data_dict[datf]["t"][increase_ind]
        # data_dict[datf]["yextrap"] += data_dict[datf]["y"][increase_ind]
        # Uncertainty propagation
        lim_ind = data_dict[datf]["lim"]
        nsamples = np_hmc_samples.shape[0]
        realization = []
        for i in range(nsamples):
            yext = sigm_fit_jax(
                np_hmc_samples[i, :], data_dict[datf]["textrap"]
            )
            yext += data_dict[datf]["y"][lim_ind]
            text = data_dict[datf]["textrap"] + data_dict[datf]["t"][lim_ind]
            if np.amax(yext) < 1:
                realization.append(yext)
        realization = np.array(realization)
        mean_real = np.mean(realization, axis=0)
        med_real = np.median(realization, axis=0)
        std84_real = np.percentile(realization, 84, axis=0)
        std16_real = np.percentile(realization, 16, axis=0)

        data_dict[datf]["textrap"] = text
        data_dict[datf]["mean_real"] = mean_real
        data_dict[datf]["med_real"] = med_real
        data_dict[datf]["std84_real"] = std84_real
        data_dict[datf]["std16_real"] = std16_real

    return data_dict


if __name__ == "__main__":
    from bird import BIRD_EARLY_PRED_DATA_DIR

    data_dict, color_files = multi_data_load(BIRD_EARLY_PRED_DATA_DIR)
    data_dict = fit_and_ext(data_dict)
    plotAllEarly(data_dict, color_files=color_files, chop=True, extrap=True)
    bayes_fit(data_dict)
    plotAllEarly_uq(data_dict, color_files=color_files)
    plt.show()
