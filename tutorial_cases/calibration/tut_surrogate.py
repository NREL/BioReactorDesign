import os

import numpy as np
import tensorflow as tf
from prettyPlot.plotting import *

from bird.calibration.param_nn import Param_NN


def simulation(mu, sigma, x):
    return (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(
        -((x - mu) ** 2) / (2 * sigma**2)
    )


def concat_np_dat(x, mu, sigma):
    assert len(mu.shape) == 1
    assert mu.shape == sigma.shape
    assert len(x.shape) == 1
    n_dat_var = len(x)
    n_sim = len(mu)
    x_ind = 0
    mu_ind = 1
    sigma_ind = 2
    n_data = n_sim * n_dat_var
    dim = 3
    dataset_x = np.zeros((n_data, dim))
    dataset_y = np.zeros((n_data,))
    for i_sim in range(n_sim):
        beg = i_sim * n_dat_var
        end = (i_sim + 1) * n_dat_var
        dataset_x[beg:end, x_ind] = x
        dataset_x[beg:end, mu_ind] = mu[i_sim]
        dataset_x[beg:end, sigma_ind] = sigma[i_sim]
        dataset_y[beg:end] = simulation(mu[i_sim], sigma[i_sim], x)

    return dataset_x, dataset_y


def make_dataset(n_sim, n_dat_var=32):
    parameter_names = ["mu", "sigma"]
    variable_names = ["x"]

    input_names = variable_names + parameter_names

    dim = len(input_names)
    dim_var = len(variable_names)
    dim_par = len(parameter_names)

    mu = np.random.uniform(0.01, 0.9, n_sim)
    sigma = np.random.uniform(0.01, 0.9, n_sim)
    x = np.linspace(1e-6, 1 - 1e-6, n_dat_var)

    return concat_np_dat(x, mu, sigma)


def plot_loss(loss_dict):
    fig = plt.figure()
    plt.plot(
        loss_dict["epoch"],
        loss_dict["train_loss"],
        color="k",
        linewidth=3,
        label="train",
    )
    plt.plot(
        loss_dict["epoch"],
        loss_dict["val_loss"],
        color="b",
        linewidth=3,
        label="test",
    )
    pretty_labels("Epoch", "Loss", 16)
    pretty_legend()
    ax = plt.gca()
    ax.set_yscale("log")
    plt.tight_layout()


def plot_test(nn, n_sim_test=5):
    # test
    mu = np.random.uniform(0.01, 0.9, n_sim_test)
    sigma = np.random.uniform(0.01, 0.9, n_sim_test)
    n_dat_var = 64
    x = np.linspace(1e-6, 1 - 1e-6, n_dat_var)

    data_test_x, data_test_y = concat_np_dat(x, mu, sigma)
    y_pred = nn.pred(data_test_x[:, 0], data_test_x[:, 1:])

    fig, axs = plt.subplots(1, n_sim_test, figsize=(4 * n_sim_test, 4))
    for i_sim_test in range(n_sim_test):
        beg = i_sim_test * n_dat_var
        end = (i_sim_test + 1) * n_dat_var
        axs[i_sim_test].plot(x, data_test_y[beg:end], label="true")
        axs[i_sim_test].plot(x, y_pred[beg:end], label="pred")
        pretty_labels(
            "",
            "",
            16,
            ax=axs[i_sim_test],
            title=rf"$\mu = {mu[i_sim_test]:.2f}$ $\sigma = {sigma[i_sim_test]:.2f}$",
            grid=False,
        )
        if i_sim_test == n_sim_test - 1:
            pretty_legend(ax=axs[i_sim_test])


if __name__ == "__main__":
    n_par = 2
    n_var = 1
    np.random.seed(0)
    tf.random.set_seed(0)
    x, y = make_dataset(200, 64)
    np.savez("data_raw.npz", x=x.astype("float32"), y=y.astype("float32"))

    nn = Param_NN(
        input_dim=n_var + n_par,
        output_dim=1,
        units=[10, 10, 10, 5],
        activation="tanh",
        final_activation="elu",
        model_folder="Modeltmp",
        log_loss_folder="Logtmp",
    )

    nn.train(
        learningRateModel=1e-3,
        batch_size=128,
        nEpochs=2000,
        data_file="data_raw.npz",
    )

    # Plot loss
    plot_loss(nn.get_loss_dat())

    # Plot test
    plot_test(nn, n_sim_test=5)
    plt.show()
