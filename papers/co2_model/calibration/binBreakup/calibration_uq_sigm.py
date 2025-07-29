import json
import os

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

parser = argparse.ArgumentParser(description="BCR NN cal")
parser.add_argument(
    "-id",
    "--experiment_id",
    type=int,
    metavar="",
    required=True,
    help="id of the experiment (either 17 or 19)",
    default=None,
)
parser.add_argument(
    "-qoi",
    "--qoi",
    type=str,
    metavar="",
    required=True,
    help="qoi (either co2 or gh)",
    default=None,
)

args, unknown = parser.parse_known_args()

nexp = args.experiment_id
valexp = args.qoi


def load_observation_data():
    if valexp.lower() == "co2":
        A = np.load("../../validation/refData/data/val2_data.npz")[
            f"xco2_exp{nexp}"
        ].astype("float32")
    elif valexp.lower() == "gh":
        A = np.load("../../validation/refData/data/val2_data.npz")[
            f"gh_exp{nexp}"
        ].astype("float32")
    data_z = A[:, 1]
    data_y = A[:, 0]
    return data_z, data_y


data_z, data_y = load_observation_data()
z = np.reshape(data_z, (data_z.shape[0], 1))
z_tens = tf.convert_to_tensor(z, dtype=tf.dtypes.float32)
ones_tf32 = tf.ones(tf.shape(z_tens[:, 0]), dtype=tf.dtypes.float32)
nn_kwargs = {
    "input_dim": 4,
    "output_dim": 1,
    "units": [10, 20, 10, 5],
    "activation": "tanh",
}
train_kwargs = {"learningRateModel": 1e-3, "batch_size": 32, "nEpochs": 1000}
model_folder = f"Model_{valexp.lower()}_{nexp}"
if valexp.lower() == "co2":
    xLabel = r"$X_{CO_2}$"
else:
    xLabel = r"Gas Holdup"
yLabel = r"z [m]"

nn = BCR_NN(
    **nn_kwargs,
    model_folder=model_folder,
    weight_file=os.path.join(model_folder, "best.h5"),
)
scale_z_min = joblib.load(
    os.path.join(nn.model_folder, "scaler_z.mod")
).data_min_
scale_z_max = joblib.load(
    os.path.join(nn.model_folder, "scaler_z.mod")
).data_max_
scale_par_min = joblib.load(
    os.path.join(nn.model_folder, "scaler_par.mod")
).data_min_
scale_par_max = joblib.load(
    os.path.join(nn.model_folder, "scaler_par.mod")
).data_max_
scale_y_mean = joblib.load(os.path.join(nn.model_folder, "scaler_y.mod")).mean_
scale_y_scale = joblib.load(
    os.path.join(nn.model_folder, "scaler_y.mod")
).scale_


@tf.function
def forward(p):
    par_tens = tf.stack([p[i] * ones_tf32 for i in range(len(p))], axis=1)
    par_tens_resc = scale_par(par_tens, scale_par_min, scale_par_max)
    z_tens_resc = scale_z(z_tens, scale_z_min, scale_z_max)
    out_resc = nn.model([z_tens_resc, par_tens_resc])
    out = unscale_y(out_resc, scale_y_mean, scale_y_scale)
    return out[:, 0]


p = np.random.normal(size=(3,)).astype(np.float32)
jax_func, jax_params = tf2jax.convert(forward, np.zeros_like(p))
jax_forw = jax.jit(jax_func)

mcmc_method = "hmc"
num_warmup = 10000
num_samples = 15000
max_sigma = 1


def bayes_step(y=None):
    surfaceTension = numpyro.sample(
        "surfaceTension", dist.Uniform(0.065, 0.075)
    )
    coal_eff = numpyro.sample("coal_eff", dist.Uniform(0.05, 20))
    breakup_eff = numpyro.sample("breakup_eff", dist.Uniform(0.05, 20))
    y_err = numpyro.sample("err", dist.Uniform(0.001, 0.1))
    y_model, _ = jax_func(
        jax_params, jnp.array([surfaceTension, coal_eff, breakup_eff])
    )
    std_obs = jnp.ones(y_model.shape[0]) * y_err
    numpyro.sample("obs", dist.Normal(y_model, std_obs), obs=y)


def mcmc_iter(mcmc_method="HMC"):
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    # Guess
    theta = []
    theta.append(np.random.uniform(0.065, 0.075))
    theta.append(np.random.uniform(0.05, 20))
    theta.append(np.random.uniform(0.05, 20))
    theta.append(np.random.uniform(0.001, 0.1))

    # Hamiltonian Monte Carlo (HMC) with no u turn sampling (NUTS)
    if mcmc_method.lower() == "hmc":
        kernel = NUTS(bayes_step, target_accept_prob=0.9)
    elif mcmc_method.lower() == "sa":
        kernel = SA(bayes_step)
    else:
        raise NotImplementedError(f"MCMC method {mcmc_method} unrecognized")

    mcmc = MCMC(
        kernel,
        num_chains=1,
        num_warmup=num_warmup,
        num_samples=num_samples,
        jit_model_args=True,
    )
    mcmc.run(rng_key_, y=data_y)
    mcmc.print_summary()

    # Draw samples
    mcmc_samples = mcmc.get_samples()
    labels = list(mcmc_samples.keys())
    nsamples = len(mcmc_samples[labels[0]])
    nparams = len(labels)
    np_mcmc_samples = np.zeros((nsamples, nparams))
    labels_np = ["surfaceTension", "coal_eff", "breakup_eff", "err"]
    for ilabel, label in enumerate(labels):
        for ipar, name in enumerate(
            ["surfaceTension", "coal_eff", "breakup_eff", "err"]
        ):
            if label == name:
                nplabel = labels_np.index(name)
        np_mcmc_samples[:, nplabel] = np.array(mcmc_samples[label])

    # Uncertainty propagation
    nsamples = np_mcmc_samples.shape[0]
    realization = []
    for i in range(nsamples):
        y = forward(np_mcmc_samples[i, :3].astype("float32"))
        realization.append(y)
    realization = np.array(realization)

    # mean_real = np.mean(realization, axis=0)
    # min_real = np.min(realization, axis=0)
    # max_real = np.max(realization, axis=0)
    min_real = np.percentile(realization, 2.5, axis=0)
    max_real = np.percentile(realization, 97.5, axis=0)

    results = {
        "samples": np_mcmc_samples,
        "labels_np": labels_np,
        "labels": labels,
    }

    return True, results


_, results = mcmc_iter(mcmc_method=mcmc_method)

sigma = np.mean(results["samples"][:, -1])
with open("sigma.log", "a+") as f:
    f.write(f"{args.experiment_id} {args.qoi} {sigma}\n")

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
ranges.append((0.001, 0.1))
truths = None

labels_np_disp = labels_np.copy()
ind_surftens = labels_np_disp.index("surfaceTension")
labels_np_disp[ind_surftens] = r"$\sigma$ [N/m]"
ind_coal = labels_np_disp.index("coal_eff")
labels_np_disp[ind_coal] = r"Coalescence Eff."
ind_break = labels_np_disp.index("breakup_eff")
labels_np_disp[ind_break] = r"Breakup Eff."
ind_break = labels_np_disp.index("err")
labels_np_disp[ind_break] = r"$\sigma$"
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

plt.savefig(os.path.join(figureFolder, f"corner_{valexp}_{nexp}.png"))
plt.savefig(os.path.join(figureFolder, f"corner_{valexp}_{nexp}.eps"))
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
plt.savefig(os.path.join(figureFolder, f"seq_{valexp}_{nexp}.png"))
plt.savefig(os.path.join(figureFolder, f"seq_{valexp}_{nexp}.eps"))
plt.close()
# plt.savefig(os.path.join(figureFolder, f"seq.png"))
# plt.close()


# Uncertainty propagation
rangez = np.reshape(
    np.linspace(np.amin(data_z), np.amax(data_z), 250), (250, 1)
)
z_tens = tf.convert_to_tensor(rangez, dtype=tf.dtypes.float32)
ones_tf32 = tf.ones(tf.shape(rangez[:, 0]), dtype=tf.dtypes.float32)


@tf.function
def forward_range(p, rangez):
    par = tf.stack([p[i] * ones_tf32 for i in range(len(p))], axis=1)
    par = scale_par(par, scale_par_min, scale_par_max)
    z_tens_resc = scale_z(z_tens, scale_z_min, scale_z_max)
    out = nn.model([z_tens_resc, par])
    out = unscale_y(out, scale_y_mean, scale_y_scale)
    return out[:, 0]


nsamples = np_mcmc_samples.shape[0]
print("Num samples = ", nsamples)
realization = []
for i in range(nsamples):
    y = forward_range(
        np_mcmc_samples[i, :3].astype("float32"), rangez.astype("float32")
    )
    realization.append(y)
realization = np.array(realization)

mean_real = np.mean(realization, axis=0)
min_real = np.min(realization, axis=0)
max_real = np.max(realization, axis=0)
std97_5_real = np.percentile(realization, 97.5, axis=0)
std2_5_real = np.percentile(realization, 2.5, axis=0)

fig = plt.figure()
plt.plot(data_y, data_z, "o", color="r", markersize=7, label="Data")
plt.plot(data_y - 2 * sigma, data_z, "--", color="r")
plt.plot(
    data_y + 2 * sigma,
    data_z,
    "--",
    color="r",
    label="95% Exp. confidence interval",
)
# ax.grid()
plt.plot(mean_real, rangez, color="k", linewidth=3, label="mean degradation")
plt.plot(
    std97_5_real,
    rangez,
    "--",
    color="k",
    linewidth=3,
    label="95% Model confidence interval",
)
plt.plot(std2_5_real, rangez, "--", color="k", linewidth=3)
pretty_labels(
    xLabel,
    yLabel,
    20,
    title=f"Exp uncertainty = {sigma:.4g}",
    fontname="Times New Roman",
)
# pretty_legend()
plt.savefig(os.path.join(figureFolder, f"forw_{valexp}_{nexp}.png"))
plt.savefig(os.path.join(figureFolder, f"forw_{valexp}_{nexp}.eps"))
plt.close()


# plt.show()
