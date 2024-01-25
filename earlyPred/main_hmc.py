import numpy as np
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import corner
import os
from prettyPlot.plotting import *


dataRoot = 'data'
color_files = ["k", "r", "b"]
data_files = ["240rpm_1e8", "150rpm_1e8", "150rpm_5e7"]
data_arr = {}

for idat, datf in enumerate(data_files):
    filename = os.path.join(dataRoot, datf)
    A = np.loadtxt(filename)
    data_arr[datf] = {}
    data_arr[datf]["t"] = A[:,0]
    data_arr[datf]["y"] = A[:,5] / (A[:,4]*16/44 + A[:,5])
    increase_ind_arr = np.argwhere(np.diff(data_arr[datf]["y"]) > 0)
    increase_ind = increase_ind_arr[np.argwhere(data_arr[datf]["t"][increase_ind_arr] > 10)[0][0]][0]
    data_arr[datf]["lim"] = increase_ind
    

    y_fit = data_arr[datf]["y"][increase_ind:] - data_arr[datf]["y"][increase_ind]
    t_fit = data_arr[datf]["t"][increase_ind:] - data_arr[datf]["t"][increase_ind]
    data_arr[datf]["textrap"] = np.linspace(0,600,600)
    data_arr[datf]["tfit"] = t_fit
    data_arr[datf]["yfit"] = y_fit

maxExp = 20
num_warmup = 1000
num_samples = 500

# Guess
theta = [0.0369432,  1.50551883, 0.38822064, 0.01314411, 0.98011988, 0.45693512, 1e-4]

def SREModel(theta, x):
    a, b, M, c, d, N, sigma = theta
    F =  2*M * (0.5 - 1/(1+jnp.exp(jnp.clip((a*x)**b,a_min=None, a_max=maxExp))))
    F += 2*N * (0.5 - 1/(1+jnp.exp(jnp.clip((c*x)**d,a_min=None, a_max=maxExp))))
    return F

def bayes_step(x, y=None, y_err=0.1):
    # define parameters (incl. prior ranges)
    a = numpyro.sample('a', dist.Uniform(0, 0.05))
    b = numpyro.sample('b', dist.Uniform(0, 1.7))
    M = numpyro.sample('M', dist.Uniform(0, 1))
    c = numpyro.sample('c', dist.Uniform(0, 0.015))
    d = numpyro.sample('d', dist.Uniform(0.5, 1.3))
    N = numpyro.sample('N', dist.Uniform(0, 1))
    sigma = numpyro.sample('sigma', dist.Uniform(1e-2, 10))
    
    # implement the model
    # needs jax numpy for differentiability here
    y_model = SREModel([a,b,M,c,d,N,sigma], x)

    # notice that we clamp the outcome of this sampling to the observation y
    numpyro.sample('obs', dist.Normal(y_model, sigma), obs=y)

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

for idat, datf in enumerate(data_files):
    # Hamilton Markov Chain (HMC) with no u turn sampling (NUTS)
    kernel = NUTS(bayes_step, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key_, x=data_arr[datf]["tfit"], y=data_arr[datf]["yfit"])
    mcmc.print_summary()

    # Draw samples
    hmc_samples = mcmc.get_samples()
    labels = list(hmc_samples.keys())
    nsamples = len(hmc_samples[labels[0]])
    nparams = len(labels)
    np_hmc_samples = np.zeros((nsamples, nparams))
    for ilabel, label in enumerate(labels):
        if label == 'a':
            nplabel=0
        if label == 'b':
            nplabel=1
        if label == 'M':
            nplabel=2
        if label == 'c':
            nplabel=3
        if label == 'd':
            nplabel=4
        if label == 'N':
            nplabel=5
        if label == 'sigma':
            nplabel=6
        np_hmc_samples[:,nplabel] = np.array(hmc_samples[label])
    
    ## Post process
    #fig = corner.corner(np_hmc_samples, truths=theta, labels=labels)
    #
    #
    ## Convergence
    #fig, axes = plt.subplots(nparams, sharex=True)
    #for i in range(nparams):
    #    ax = axes[i]
    #    ax.plot(np_hmc_samples[:, i], "k", alpha=0.3, rasterized=True)
    #    ax.set_ylabel(labels[i])
    
    #data_arr[datf]["yextrap"] = f(data_arr[datf]["textrap"], *popt)
    #data_arr[datf]["textrap"] += data_arr[datf]["t"][increase_ind]
    #data_arr[datf]["yextrap"] += data_arr[datf]["y"][increase_ind]
    # Uncertainty propagation
    nsamples = np_hmc_samples.shape[0]
    realization = []
    for i in range(nsamples):
        yext = SREModel(np_hmc_samples[i,:], data_arr[datf]["textrap"])
        yext += data_arr[datf]["y"][increase_ind] 
        text = data_arr[datf]["textrap"] + data_arr[datf]["t"][increase_ind]
        if np.amax(yext) < 1:
           realization.append(yext)
    realization = np.array(realization)
    mean_real = np.mean(realization, axis=0)
    med_real = np.median(realization, axis=0)
    std90_real = np.percentile(realization, 84, axis=0)
    std10_real = np.percentile(realization, 16, axis=0)
         
    data_arr[datf]["textrap"] = text
    data_arr[datf]["mean_real"] = mean_real
    data_arr[datf]["med_real"] = med_real
    data_arr[datf]["std90_real"] = std90_real
    data_arr[datf]["std10_real"] = std10_real



fig = plt.figure()
for idat, datf in enumerate(data_files):
    color=color_files[idat]
    text = data_arr[datf]["textrap"]
    t = data_arr[datf]["t"]
    y = data_arr[datf]["y"] * 100
    med_real = data_arr[datf]["med_real"] * 100
    mean_real = data_arr[datf]["mean_real"] * 100
    std10_real = data_arr[datf]["std10_real"] * 100
    std90_real = data_arr[datf]["std90_real"] * 100

    plt.plot(text, med_real, color=color, linewidth=3)
    plt.fill_between(text, std10_real, std90_real, color=color, alpha=0.3)
    #plt.plot(text, std90_real, '--', color=color, linewidth=3)
    #plt.plot(text, std10_real, '--', color=color, linewidth=3)
    plt.plot(t, y, 'o', color=color, markersize=5, linewidth=3, label=f"{datf} y={med_real[-1]:.2f}+/-{(std90_real[-1]-std10_real[-1])/2:.2f}" )
    pretty_labels('time [s]', 'yield [%]', 14)
    pretty_legend()
plt.show()

