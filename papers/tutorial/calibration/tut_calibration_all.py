import argparse
import json
import os
import sys

import corner
import jax
import jax.numpy as jnp
import jax.random as random
import joblib
import keras
import numpy as np
import numpyro
import numpyro.distributions as dist
import tensorflow as tf
import tf2jax
from keras import layers
from numpyro.infer import MCMC, NUTS
from post_cal import post_process_cal
from prettyPlot.plotting import *

from bird.calibration.param_nn import Param_NN
from bird.calibration.scaling import *

parser = argparse.ArgumentParser(description="Calibration Interface")
parser.add_argument(
    "-useNN",
    "--useNN",
    action="store_true",
    help="Use neural net surrogate",
)
parser.add_argument(
    "-alpha",
    "--alpha",
    type=float,
    metavar="",
    required=False,
    help="alpha parameter in the beta distributio",
    default=5,
)
parser.add_argument(
    "-beta",
    "--beta",
    type=float,
    metavar="",
    required=False,
    help="beta parameter in the beta distributio",
    default=5,
)

args, unknown = parser.parse_known_args()


def observation_data(alpha=2, beta=5):
    from scipy.special import factorial, gamma

    x = np.random.uniform(1e-6, 1 - 1e-6, 128)
    x = x[np.argsort(x)]
    B = gamma(alpha) * gamma(beta) / gamma(alpha + beta)
    y = x ** (alpha - 1) * (1 - x) ** (beta - 1) / B
    return x, y


data_x, data_y = observation_data(alpha=args.alpha, beta=args.beta)
rangex = np.reshape(
    np.linspace(np.amin(data_x), np.amax(data_x), 250), (250, 1)
)
if args.useNN:
    print("INFO: Using NN surrogate")
    x = np.reshape(data_x, (data_x.shape[0], 1))
    x_tens = tf.convert_to_tensor(x, dtype=tf.dtypes.float32)
    ones_tf32 = tf.ones(tf.shape(x_tens[:, 0]), dtype=tf.dtypes.float32)
    nn_kwargs = {
        "input_dim": 3,
        "output_dim": 1,
        "units": [10, 10, 10, 5],
        "activation": "tanh",
        "final_activation": "elu",
        "model_folder": "Modeltmp",
    }
    nn = Param_NN(
        **nn_kwargs, weight_file=os.path.join("Modeltmp", "best.weights.h5")
    )
    scale_x_min = joblib.load(
        os.path.join(nn.model_folder, "scaler_z.mod")
    ).data_min_
    scale_x_max = joblib.load(
        os.path.join(nn.model_folder, "scaler_z.mod")
    ).data_max_
    scale_par_min = joblib.load(
        os.path.join(nn.model_folder, "scaler_par.mod")
    ).data_min_
    scale_par_max = joblib.load(
        os.path.join(nn.model_folder, "scaler_par.mod")
    ).data_max_
    scale_y_mean = joblib.load(
        os.path.join(nn.model_folder, "scaler_y.mod")
    ).mean_
    scale_y_scale = joblib.load(
        os.path.join(nn.model_folder, "scaler_y.mod")
    ).scale_

    @tf.function
    def forwardNN(p):
        par_tens = tf.stack([p[i] * ones_tf32 for i in range(len(p))], axis=1)
        par_tens_resc = scale_par(par_tens, scale_par_min, scale_par_max)
        x_tens_resc = scale_x(x_tens, scale_x_min, scale_x_max)
        out_resc = nn.model([x_tens_resc, par_tens_resc])
        out = unscale_y(out_resc, scale_y_mean, scale_y_scale)
        return out[:, 0]

    p = np.random.normal(size=(2,)).astype(np.float32)
    jax_func, jax_params = tf2jax.convert(forwardNN, np.zeros_like(p))

    def forward(p):
        return jax_func(jax_params, p)[0]

    # Uncertainty propagation
    rangex_tens = tf.convert_to_tensor(rangex, dtype=tf.dtypes.float32)
    rangeones_tf32 = tf.ones(tf.shape(rangex[:, 0]), dtype=tf.dtypes.float32)

    @tf.function
    def forward_range(p):
        par = tf.stack([p[i] * rangeones_tf32 for i in range(len(p))], axis=1)
        par = scale_par(par, scale_par_min, scale_par_max)
        rangex_tens_resc = scale_x(rangex_tens, scale_x_min, scale_x_max)
        out = nn.model([rangex_tens_resc, par])
        out = unscale_y(out, scale_y_mean, scale_y_scale)
        return out[:, 0]

else:
    print("INFO: Using true function")

    def forward(p):
        return (1 / jnp.sqrt(2 * jnp.pi * p[1] ** 2)) * jnp.exp(
            -((jnp.array(data_x) - p[0]) ** 2) / (2 * p[1] ** 2)
        )

    def forward_range(p):
        return (1 / jnp.sqrt(2 * jnp.pi * p[1] ** 2)) * jnp.exp(
            -((jnp.array(rangex) - p[0]) ** 2) / (2 * p[1] ** 2)
        )


def bayes_step(y=None, y_err=0.1):
    mu_d = numpyro.sample("mu_d", dist.Uniform(0.01, 0.9))
    sigma_d = numpyro.sample("sigma_d", dist.Uniform(0.01, 0.9))
    y_model = forward(jnp.array([mu_d, sigma_d]))
    std_obs = jnp.ones(y_model.shape[0]) * y_err
    numpyro.sample("obs", dist.Normal(y_model, std_obs), obs=y)


def mcmc_iter(y_err=0.1, mcmc_method="HMC"):
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    # Guess
    theta = []
    theta.append(np.random.uniform(0.01, 0.9))
    theta.append(np.random.uniform(0.01, 0.9))

    # Hamiltonian Monte Carlo (HMC) with no u turn sampling (NUTS)
    if mcmc_method.lower() == "hmc":
        kernel = NUTS(bayes_step, target_accept_prob=0.9)
    elif mcmc_method.lower() == "sa":
        kernel = SA(bayes_step)
    else:
        sys.exit(f"MCMC method {mcmc_method} unrecognized")
    num_warmup = 1000
    num_samples = 1500
    mcmc = MCMC(
        kernel,
        num_chains=1,
        num_warmup=num_warmup,
        num_samples=num_samples,
        jit_model_args=True,
    )
    mcmc.run(rng_key_, y=data_y, y_err=y_err)
    mcmc.print_summary()

    # Draw samples
    mcmc_samples = mcmc.get_samples()
    labels = list(mcmc_samples.keys())
    nsamples = len(mcmc_samples[labels[0]])
    nparams = len(labels)
    np_mcmc_samples = np.zeros((nsamples, nparams))
    labels_np = ["mu_d", "sigma_d"]
    for ilabel, label in enumerate(labels):
        for ipar, name in enumerate(["mu_d", "sigma_d"]):
            if label == name:
                nplabel = labels_np.index(name)
        np_mcmc_samples[:, nplabel] = np.array(mcmc_samples[label])

    # Uncertainty propagation
    nsamples = np_mcmc_samples.shape[0]
    realization = []
    for i in range(nsamples):
        y = forward(
            jnp.array(
                [
                    np_mcmc_samples[i, 0].astype("float32"),
                    np_mcmc_samples[i, 1].astype("float32"),
                ]
            )
        )
        realization.append(y)
    realization = np.array(realization)

    min_real = np.percentile(realization, 2.5, axis=0)
    max_real = np.percentile(realization, 97.5, axis=0)

    results = {
        "samples": np_mcmc_samples,
        "labels_np": labels_np,
        "labels": labels,
    }

    true_m95 = data_y - 2 * y_err
    true_p95 = data_y + 2 * y_err

    if np.amax(true_m95 - min_real) > 0 or np.amin(true_p95 - max_real) < 0:
        print(
            f" Increase STD  {np.amax(true_m95 - min_real)} - {np.amin(true_p95 - max_real)}"
        )
        return False, results
    else:
        return True, results


min_sigma = 1e-4
max_sigma = 1
guess_sigma = 0.07
sigma = guess_sigma

for iteration_sigma in range(10):
    print(f"Doing sigma = {sigma:.3g}")
    reduce_sigma, results = mcmc_iter(y_err=sigma, mcmc_method="hmc")
    if reduce_sigma:
        max_sigma = sigma
        sigma = sigma - (sigma - min_sigma) / 2
    else:
        min_sigma = sigma
        sigma = sigma + (max_sigma - sigma) / 2

if not reduce_sigma:
    reduce_sigma, results = mcmc_iter(y_err=max_sigma, mcmc_method="hmc")

np_mcmc_samples = results["samples"]
labels_np = results["labels_np"]
labels = results["labels"]
nparams = len(labels)
np.savez(
    os.path.join("samp.npz"),
    samples=np_mcmc_samples,
    labels_np=labels_np,
    labels=labels,
)

post_process_cal(
    labels_np,
    labels,
    np_mcmc_samples,
    rangex,
    forward_range,
    data_x,
    data_y,
    sigma,
)
