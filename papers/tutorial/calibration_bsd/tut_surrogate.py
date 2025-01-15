import os

import numpy as np
import tensorflow as tf
from prettyPlot.plotting import *

from bird.calibration.param_nn import Param_NN


def make_dataset(data_folder):
    parameter_names = ["beff_fact", "ceff"]
    variable_names = ["x"]

    input_names = variable_names + parameter_names

    dim = len(input_names)
    dim_var = len(variable_names)
    dim_par = len(parameter_names)

    dirl = os.listdir(data_folder)
    tmp = np.load(os.path.join(data_folder, "target.npz"))
    xlen = len(tmp["x"])
    x_ind = 0
    beff_fact_ind = 1
    ceff_ind = 2
    count = 0
    for ifile, file in enumerate(dirl):
        if file.startswith("dataset"):
            count += 1
            tmp = np.load(os.path.join(data_folder, file))
            ndata = len(tmp["ceff"])
            tmp_dataset_x = np.zeros((xlen * ndata, 3))
            tmp_dataset_y = np.zeros((xlen * ndata,))
            sim_count = 0
            for i_sim in range(ndata):
                # beg = sim_count * xlen
                # end = (sim_count + 1) * xlen
                beg = i_sim * xlen
                end = (i_sim + 1) * xlen
                if abs(np.mean(tmp["y"][i_sim, :]) - 1) > 1e-6:
                    tmp_dataset_x[beg:end, x_ind] = tmp["x"]
                    tmp_dataset_x[beg:end, beff_fact_ind] = tmp["beff_fact"][
                        i_sim
                    ]
                    tmp_dataset_x[beg:end, ceff_ind] = tmp["ceff"][i_sim]
                    tmp_dataset_y[beg:end] = tmp["y"][i_sim, :]
                    sim_count += 1
                else:
                    tmp_dataset_x[beg:end, x_ind] = tmp["x"]
                    tmp_dataset_x[beg:end, beff_fact_ind] = tmp["beff_fact"][
                        i_sim
                    ]
                    tmp_dataset_x[beg:end, ceff_ind] = tmp["ceff"][i_sim]
                    tmp_dataset_y[beg:end] = tmp["y"][i_sim, :]
            print(sim_count)
            if count == 1:
                # dataset_x = tmp_dataset_x[:sim_count*xlen]
                # dataset_y = tmp_dataset_y[:sim_count*xlen]
                dataset_x = tmp_dataset_x[: ndata * xlen]
                dataset_y = tmp_dataset_y[: ndata * xlen]
            else:
                # dataset_x = np.vstack((dataset_x,tmp_dataset_x[:sim_count*xlen]))
                # dataset_y = np.hstack((dataset_y, tmp_dataset_y[:sim_count*xlen]))
                dataset_x = np.vstack(
                    (dataset_x, tmp_dataset_x[: ndata * xlen])
                )
                dataset_y = np.hstack(
                    (dataset_y, tmp_dataset_y[: ndata * xlen])
                )
            print(f"file {file} {dataset_x.shape}")

    return dataset_x, dataset_y


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


if __name__ == "__main__":
    n_par = 2
    n_var = 1
    np.random.seed(0)
    tf.random.set_seed(0)
    x, y = make_dataset("data")
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
        nEpochs=3000,
        data_file="data_raw.npz",
    )

    # Plot loss
    plot_loss(nn.get_loss_dat())

    plt.show()
