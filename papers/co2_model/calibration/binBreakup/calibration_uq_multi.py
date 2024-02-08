import json
import os
import sys

import corner
import jax
import jax.numpy as jnp
import jax.random as random
import keras
import numpy as np
import numpyro
import numpyro.distributions as dist
import tensorflow as tf
import tf2jax
from keras import layers
from numpyro.infer import MCMC, NUTS
from prettyPlot.plotting import *

root = "."

from bcr_nn_list import *


def scale_z(inp, min_, max_):
    return (inp - min_) / (max_ - min_)


def scale_par(inp, min_, max_):
    return scale_z(inp, min_, max_)


def scale_y(inp, mean_, scale_):
    return (inp - mean_) / (max_ - min_)


def unscale_z(inp, min_, max_):
    return inp * (max_ - min_) + min_


def unscale_par(inp, min_, max_):
    return unscale_z(inp, min_, max_)


def unscale_y(inp, mean_, scale_):
    return inp * scale_ + mean_


import argparse

xLabel_list = [r"$X_{CO_2}$", r"$X_{CO_2}$", r"Gas Holdup", r"Gas Holdup"]
yLabel = r"z [m]"
obs_name_list = ["xco2_exp17", "xco2_exp19", "gh_exp17", "gh_exp19"]
mod_list = ["Model_co2_17", "Model_co2_19", "Model_gh_17", "Model_gh_19"]
with open("sigma.log", "r") as f:
    lines = f.readlines()
    sigmas = {}
    for line in lines:
        tmp = line.split()
        sigmas[f"{tmp[0]}_{tmp[1]}"] = np.float32(tmp[2])

err_list = [
    sigmas["17_co2"],
    sigmas["19_co2"],
    sigmas["17_gh"],
    sigmas["19_gh"],
]

nn_kwargs = {
    "input_dim": 4,
    "output_dim": 1,
    "units": [10, 20, 10, 5],
    "activation": "tanh",
}
train_kwargs = {"learningRateModel": 1e-3, "batch_size": 32, "nEpochs": 1000}


def load_observation_data(obs_name_list):
    val_data = np.load("../../validation/refData/data/val2_data.npz")
    obs_dict = {}
    for obs_name in obs_name_list:
        A = val_data[obs_name].astype("float32")
        data_z = A[:, 1]
        data_y = A[:, 0]
        obs_dict[obs_name] = [data_z, data_y]
    return obs_dict


obs_dict = load_observation_data(obs_name_list)
data_z_list = [obs_dict[obs_name][0] for obs_name in obs_name_list]
data_y_list = [obs_dict[obs_name][1] for obs_name in obs_name_list]

z_list = [np.reshape(data_z, (data_z.shape[0], 1)) for data_z in data_z_list]
z_tens_list = [
    tf.convert_to_tensor(z, dtype=tf.dtypes.float32) for z in z_list
]
ones_tf32_list = [
    tf.ones(tf.shape(z_tens[:, 0]), dtype=tf.dtypes.float32)
    for z_tens in z_tens_list
]
nn_list = [
    BCR_NN(
        **nn_kwargs, model_folder=mod, weight_file=os.path.join(mod, "best.h5")
    )
    for mod in mod_list
]
scale_z_min_list = [
    joblib.load(os.path.join(nn.model_folder, "scaler_z.mod")).data_min_
    for nn in nn_list
]
scale_z_max_list = [
    joblib.load(os.path.join(nn.model_folder, "scaler_z.mod")).data_max_
    for nn in nn_list
]
scale_par_min_list = [
    joblib.load(os.path.join(nn.model_folder, "scaler_par.mod")).data_min_
    for nn in nn_list
]
scale_par_max_list = [
    joblib.load(os.path.join(nn.model_folder, "scaler_par.mod")).data_max_
    for nn in nn_list
]
scale_y_mean_list = [
    joblib.load(os.path.join(nn.model_folder, "scaler_y.mod")).mean_
    for nn in nn_list
]
scale_y_scale_list = [
    joblib.load(os.path.join(nn.model_folder, "scaler_y.mod")).scale_
    for nn in nn_list
]


@tf.function
def forward(p):
    par_tens_list = [
        tf.stack([p[i] * ones_tf32 for i in range(len(p))], axis=1)
        for ones_tf32 in ones_tf32_list
    ]
    par_tens_resc_list = [
        scale_par(par_tens, scale_par_min, scale_par_max)
        for (par_tens, scale_par_min, scale_par_max) in zip(
            par_tens_list, scale_par_min_list, scale_par_max_list
        )
    ]
    z_tens_resc_list = [
        scale_z(z_tens, scale_z_min, scale_z_max)
        for (z_tens, scale_z_min, scale_z_max) in zip(
            z_tens_list, scale_z_min_list, scale_z_max_list
        )
    ]
    out_resc_list = [
        nn.model([z_tens_resc, par_tens_resc])
        for (nn, z_tens_resc, par_tens_resc) in zip(
            nn_list, z_tens_resc_list, par_tens_resc_list
        )
    ]
    out_list = [
        unscale_y(out_resc, scale_y_mean, scale_y_scale)
        for (out_resc, scale_y_mean, scale_y_scale) in zip(
            out_resc_list, scale_y_mean_list, scale_y_scale_list
        )
    ]
    return [out[:, 0] for out in out_list]


p = np.random.normal(size=(3,)).astype(np.float32)
jax_func, jax_params = tf2jax.convert(forward, np.zeros_like(p))
jax_forw = jax.jit(jax_func)

mcmc_method = "hmc"
num_warmup = 10000
num_samples = 15000
max_sigma = 1


def bayes_step(y_list=None, y_err_list=[0.1]):
    surfaceTension = numpyro.sample(
        "surfaceTension", dist.Uniform(0.065, 0.075)
    )
    coal_eff = numpyro.sample("coal_eff", dist.Uniform(0.05, 20))
    breakup_eff = numpyro.sample("breakup_eff", dist.Uniform(0.05, 20))
    y_model_list, _ = jax_func(
        jax_params, jnp.array([surfaceTension, coal_eff, breakup_eff])
    )
    std_obs = jnp.concatenate(
        [
            jnp.ones(y_model.shape[0]) * y_err
            for (y_model, y_err) in zip(y_model_list, y_err_list)
        ]
    )
    numpyro.sample(
        "obs",
        dist.Normal(jnp.concatenate(y_model_list, axis=0), std_obs),
        obs=jnp.concatenate(y_list, axis=0),
    )


def mcmc_iter(mcmc_method="HMC"):
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    # Guess
    theta = []
    theta.append(np.random.uniform(0.065, 0.075))
    theta.append(np.random.uniform(0.05, 20))
    theta.append(np.random.uniform(0.05, 20))

    # Hamiltonian Monte Carlo (HMC) with no u turn sampling (NUTS)
    if mcmc_method.lower() == "hmc":
        kernel = NUTS(bayes_step, target_accept_prob=0.9)
    elif mcmc_method.lower() == "sa":
        kernel = SA(bayes_step)
    else:
        sys.exit(f"MCMC method {mcmc_method} unrecognized")

    mcmc = MCMC(
        kernel,
        num_chains=1,
        num_warmup=num_warmup,
        num_samples=num_samples,
        jit_model_args=True,
    )
    mcmc.run(rng_key_, y_list=data_y_list, y_err_list=err_list)
    mcmc.print_summary()

    # Draw samples
    mcmc_samples = mcmc.get_samples()
    labels = list(mcmc_samples.keys())
    nsamples = len(mcmc_samples[labels[0]])
    nparams = len(labels)
    np_mcmc_samples = np.zeros((nsamples, nparams))
    labels_np = ["surfaceTension", "coal_eff", "breakup_eff"]
    for ilabel, label in enumerate(labels):
        for ipar, name in enumerate(
            ["surfaceTension", "coal_eff", "breakup_eff"]
        ):
            if label == name:
                nplabel = labels_np.index(name)
        np_mcmc_samples[:, nplabel] = np.array(mcmc_samples[label])

    results = {
        "samples": np_mcmc_samples,
        "labels_np": labels_np,
        "labels": labels,
    }

    return True, results


reduce_sigma, results = mcmc_iter(mcmc_method=mcmc_method)

np_mcmc_samples = results["samples"]
labels_np = results["labels_np"]
labels = results["labels"]
nparams = len(labels)

figureFolder = "Figures"
os.makedirs(figureFolder, exist_ok=True)

np.savez(
    os.path.join("samp.npz"),
    samples=np_mcmc_samples,
    labels_np=labels_np,
    labels=labels,
)

# Post process
ranges = []
ranges.append((0.065, 0.075))
ranges.append((0.05, 20))
ranges.append((0.05, 20))
truths = None

labels_np_disp = labels_np.copy()
ind_surftens = labels_np_disp.index("surfaceTension")
labels_np_disp[ind_surftens] = "Surf. tens. [N/m]"
ind_coal = labels_np_disp.index("coal_eff")
labels_np_disp[ind_coal] = r"Coalescence Eff."
ind_break = labels_np_disp.index("breakup_eff")
labels_np_disp[ind_break] = r"Breakup Eff."
fig = corner.corner(
    np_mcmc_samples,
    truths=truths,
    labels=labels_np_disp,
    truth_color="k",
    bins=50,
    range=ranges,
)
for ax in fig.get_axes():
    ax.tick_params(
        axis="both", labelsize=20
    )  # Customize font size, line width, and tick length
    ax.xaxis.label.set_fontweight("bold")  # Set the X-axis label to bold
    ax.yaxis.label.set_fontweight("bold")  # Set the Y-axis label to bold
    ax.xaxis.label.set_font("Times New Roman")  # Set the X-axis label to bold
    ax.yaxis.label.set_font("Times New Roman")  # Set the Y-axis label to bold
    ax.xaxis.label.set_fontsize(20)  # Set the X-axis label to bold
    ax.yaxis.label.set_fontsize(20)  # Set the Y-axis label to bold
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname("Times New Roman")
        tick.set_fontweight("bold")
for ax in fig.get_axes():
    ax.set_xlabel(ax.get_xlabel(), fontweight="bold")
    ax.set_ylabel(ax.get_ylabel(), fontweight="bold")
    plt.savefig(os.path.join(figureFolder, f"corner_all.png"))
    plt.savefig(os.path.join(figureFolder, f"corner_all.eps"))
plt.close()
# plt.savefig(os.path.join(figureFolder, f"corner.png"))
# plt.savefig(os.path.join(figureFolder, f"corner.eps"))
# plt.close()

# Convergence
fig, axes = plt.subplots(nparams, sharex=True)
for i in range(nparams):
    ax = axes[i]
    ax.plot(np_mcmc_samples[:, i], "k", alpha=0.3, rasterized=True)
    ax.set_ylabel(labels[i])
# plt.savefig(os.path.join(figureFolder, f"seq.png"))
# plt.close()


# Uncertainty propagation
rangez = np.reshape(
    np.linspace(
        np.amin(np.concatenate(data_z_list, axis=0)),
        np.amax(np.concatenate(data_z_list, axis=0)),
        250,
    ),
    (250, 1),
)
z_tens = tf.convert_to_tensor(rangez, dtype=tf.dtypes.float32)
ones_tf32 = tf.ones(tf.shape(rangez[:, 0]), dtype=tf.dtypes.float32)


@tf.function
def forward_range(p, rangez):
    par = tf.stack([p[i] * ones_tf32 for i in range(len(p))], axis=1)
    par_resc_list = [
        scale_par(par, scale_par_min, scale_par_max)
        for (scale_par_min, scale_par_max) in zip(
            scale_par_min_list, scale_par_max_list
        )
    ]
    z_tens_resc_list = [
        scale_z(z_tens, scale_z_min, scale_z_max)
        for (scale_z_min, scale_z_max) in zip(
            scale_z_min_list, scale_z_max_list
        )
    ]
    out_resc_list = [
        nn.model([z_tens_resc, par_resc])
        for (nn, z_tens_resc, par_resc) in zip(
            nn_list, z_tens_resc_list, par_resc_list
        )
    ]
    out_list = [
        unscale_y(out_resc, scale_y_mean, scale_y_scale)
        for (out_resc, scale_y_mean, scale_y_scale) in zip(
            out_resc_list, scale_y_mean_list, scale_y_scale_list
        )
    ]
    return [out[:, 0] for out in out_list]


nsamples = np_mcmc_samples.shape[0]
print("Num samples = ", nsamples)
realization_list = []
for iobs, obs_name in enumerate(obs_name_list):
    realization_list.append([])
for i in range(nsamples):
    y = forward_range(
        np_mcmc_samples[i, :].astype("float32"), rangez.astype("float32")
    )
    for iobs, obs_name in enumerate(obs_name_list):
        realization_list[iobs].append(y[iobs])
for iobs, obs_name in enumerate(obs_name_list):
    realization_list[iobs] = np.array(realization_list[iobs])

mean_real_list = [
    np.mean(realization, axis=0) for realization in realization_list
]
min_real_list = [
    np.min(realization, axis=0) for realization in realization_list
]
max_real_list = [
    np.max(realization, axis=0) for realization in realization_list
]
std97_5_real_list = [
    np.percentile(realization, 97.5, axis=0)
    for realization in realization_list
]
std2_5_real_list = [
    np.percentile(realization, 2.5, axis=0) for realization in realization_list
]


for (
    data_y,
    data_z,
    mean_real,
    std97_5_real,
    std2_5_real,
    err,
    xLabel,
    obs_name,
) in zip(
    data_y_list,
    data_z_list,
    mean_real_list,
    std97_5_real_list,
    std2_5_real_list,
    err_list,
    xLabel_list,
    obs_name_list,
):
    fig = plt.figure()
    plt.plot(data_y, data_z, "o", color="r", markersize=7, label="Data")
    plt.plot(
        data_y + 2 * err, data_z, "--", color="r", label="95% Exp confidence"
    )
    plt.plot(data_y - 2 * err, data_z, "--", color="r")
    plt.plot(
        mean_real, rangez, color="k", linewidth=3, label="mean degradation"
    )
    plt.plot(
        std97_5_real,
        rangez,
        "--",
        color="k",
        linewidth=3,
        label="95% Model confidence",
    )
    plt.plot(std2_5_real, rangez, "--", color="k", linewidth=3)
    pretty_labels(xLabel, yLabel, 20, fontname="Times New Roman")
    plt.savefig(os.path.join(figureFolder, f"forw_all_{obs_name}.png"))
    plt.savefig(os.path.join(figureFolder, f"forw_all_{obs_name}.eps"))
    plt.close()


# plt.show()
