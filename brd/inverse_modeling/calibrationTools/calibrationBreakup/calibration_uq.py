import os
import pickle
import sys

import numpy as np

sys.path.append("util")
import corner
import jax.numpy as jnp
import jax.random as random
import jax_rbf_uq
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from plotsUtil import *

root = "."

with open(os.path.join(root, "gps.pkl"), "rb") as fp:
    models = pickle.load(fp)


labels = ["surfaceTension", "henry", "coal_eff", "breakup_eff", "sigma"]

figureFolder = "Figures"
os.makedirs(figureFolder, exist_ok=True)

for model in models:
    print(f"MODEL: {model['name']}")
    os.makedirs(os.path.join(figureFolder, model["name"]), exist_ok=True)

    data = np.load(model["data"])
    x = data["x"]
    y = np.squeeze(data["y"])
    length = model["length_scale"]
    amp = model["amplitude"]
    nobs = data["nobs"]

    def forward_model(theta, length, amp, sigma):
        return jax_rbf_uq.rbf_interpolation_jax(
            x,
            y,
            jnp.reshape(jnp.array(theta), (1, -1)),
            length_scale=length,
            amplitude=amp,
            obs_noise=sigma,
        )

    num_warmup = 1000
    num_samples = 20000

    # Guess
    theta = [0.07, 1.04, 1, 1, 1e-7]

    def lnlike_fun(theta, length, amp, sigma):
        sigma = jnp.clip(sigma, a_min=1e-8, a_max=None)
        mse, std = forward_model(theta, length, amp, sigma)
        std_obs = jnp.clip(
            jnp.sqrt(jnp.clip(std, a_min=1e-16, a_max=None) / nobs),
            a_min=1e-12,
            a_max=None,
        )
        return -mse / (2 * std_obs**2) - nobs / 2 * jnp.log(
            2 * jnp.pi * std_obs**2
        )

    def bayes_step(length, amp):
        # define parameters (incl. prior ranges)
        surfaceTension = numpyro.sample(
            "surfaceTension", dist.Uniform(0.035, 0.14)
        )
        henry = numpyro.sample("henry", dist.Uniform(0.52, 2.08))
        coal_eff = numpyro.sample("coal_eff", dist.Uniform(0.1, 10))
        breakup_eff = numpyro.sample("breakup_eff", dist.Uniform(0.1, 10))
        sigma = numpyro.sample("sigma", dist.Uniform(1e-8, 1e-6))

        # implement the model
        # needs jax numpy for differentiability here
        # std_obs = jnp.ones(y.shape[0]) * 0.2
        log_like = lnlike_fun(
            [surfaceTension, henry, coal_eff, breakup_eff], length, amp, sigma
        )

        numpyro.factor("log_likelihood", log_like)

    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)

    # Hamiltonian Monte Carlo (HMC) with no u turn sampling (NUTS)
    kernel = NUTS(bayes_step, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key_, length=length, amp=amp)
    mcmc.print_summary()

    # Draw samples
    hmc_samples = mcmc.get_samples()
    labels = list(hmc_samples.keys())
    nsamples = len(hmc_samples[labels[0]])
    nparams = len(labels)
    np_hmc_samples = np.zeros((nsamples, nparams))
    labels_np = ["surfaceTension", "henry", "coal_eff", "breakup_eff", "sigma"]
    for ilabel, label in enumerate(labels):
        if label == "surfaceTension":
            nplabel = labels_np.index("surfaceTension")
        if label == "henry":
            nplabel = labels_np.index("henry")
        if label == "coal_eff":
            nplabel = labels_np.index("coal_eff")
        if label == "breakup_eff":
            nplabel = labels_np.index("breakup_eff")
        if label == "sigma":
            nplabel = labels_np.index("sigma")
        np_hmc_samples[:, nplabel] = np.array(hmc_samples[label])

    # Post process
    fig = corner.corner(np_hmc_samples, truths=theta, labels=labels_np)
    plt.savefig(os.path.join(figureFolder, model["name"], "corner.png"))
    plt.close(fig)

    # Convergence
    fig, axes = plt.subplots(nparams, sharex=True)
    for i in range(nparams):
        ax = axes[i]
        ax.plot(np_hmc_samples[:, i], "k", alpha=0.3, rasterized=True)
        ax.set_ylabel(labels[i])
    plt.close(fig)

    # plt.show()
