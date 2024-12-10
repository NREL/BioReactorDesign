import argparse
import os
import sys
import time

import joblib
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from prettyPlot.progressBar import print_progress_bar
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras import initializers, layers, optimizers, regularizers


def flexible_activation(x, activation):
    if activation is None:
        out = x
    elif activation.lower() == "swish":
        out = layers.Activation(swish_activation)(x)
    elif activation.lower() == "sigmoid":
        out = layers.Activation(activation="sigmoid")(x)
    elif activation.lower() == "softplus":
        out = layers.Activation(activation="softplus")(x)
    elif activation.lower() == "tanh":
        out = layers.Activation(activation="tanh")(x)
    elif activation.lower() == "elu":
        out = layers.Activation(activation="elu")(x)
    elif activation.lower() == "selu":
        out = layers.Activation(activation="selu")(x)
    elif activation.lower() == "gelu":
        out = layers.Activation(activation="gelu")(x)
    elif activation.lower() == "relu":
        out = layers.Activation(activation="relu")(x)
    elif activation.lower() == "leakyrelu":
        out = layers.LeakyReLU()(x)
    else:
        sys.exit(f"ERROR: unknown activation {activation}")
    return out


def singleLayer(x, n_units, activation):
    out = layers.Dense(n_units)(x)
    out = flexible_activation(out, activation)
    return out


def makeModel(input_dim_par, output_dim, units, activation, final_activation):
    # inputs
    input_z = layers.Input(shape=(1,), name="input_z")
    input_par = layers.Input(shape=(input_dim_par,), name="input_par")
    input_z_par = layers.concatenate([input_z, input_par], name="input_z_par")
    x = input_z_par
    for unit in units:
        x = singleLayer(x, unit, activation)
    output = layers.Dense(output_dim)(x)
    output = flexible_activation(output, final_activation)
    model = keras.Model([input_z, input_par], output)
    return model


class Param_NN(keras.Model):
    def __init__(
        self,
        input_dim=None,
        output_dim=None,
        units=None,
        activation=None,
        final_activation=None,
        batch_size=None,
        nEpochs=None,
        weight_file=None,
        model_folder="Model",
        log_loss_folder="Log",
        np_data_file=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.np_data_file = np_data_file
        if input_dim is None or output_dim is None:
            input_dim, output_dim = self.get_dim_from_data(np_data_file)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_dim_par = self.input_dim - 1

        self.model = makeModel(
            self.input_dim_par,
            self.output_dim,
            units,
            activation,
            final_activation,
        )
        if weight_file is not None:
            self.load(weight_file)

        self.batch_size = batch_size
        self.nEpochs = nEpochs

        self.model_folder = model_folder
        self.log_loss_folder = log_loss_folder

    def get_dim_from_data(self, data_file):
        tmp_dat = np.load(data_file)
        if len(tmp_dat["x"].shape) == 1:
            input_dim = 1
        else:
            input_dim = tmp_dat["x"].shape[1]
        if len(tmp_dat["y"].shape) == 1:
            output_dim = 1
        else:
            output_dim = tmp_dat["y"].shape[1]
        return input_dim, output_dim

    def make_data(self, np_file=None, scaler_folder="."):
        # Raw data
        if np_file is None:
            np_file = self.np_data_file

        tmp = np.load(np_file)
        z_dat = np.float32(np.reshape(tmp["x"][:, 0], (-1, 1)))
        par_dat = np.float32(tmp["x"][:, 1:])
        y_dat = np.float32(np.reshape(tmp["y"], (-1, 1)))

        # Scaler
        scaler_z = MinMaxScaler()
        scaler_par = MinMaxScaler()
        scaler_y = StandardScaler()
        scaler_z.fit(z_dat)
        scaler_par.fit(par_dat)
        scaler_y.fit(y_dat)
        joblib.dump(scaler_z, os.path.join(scaler_folder, "scaler_z.mod"))
        joblib.dump(scaler_par, os.path.join(scaler_folder, "scaler_par.mod"))
        joblib.dump(scaler_y, os.path.join(scaler_folder, "scaler_y.mod"))

        # Split and shuffle
        (
            z_train,
            z_test,
            par_train,
            par_test,
            y_train,
            y_test,
        ) = train_test_split(
            scaler_z.transform(z_dat),
            scaler_par.transform(par_dat),
            scaler_y.transform(y_dat),
            test_size=0.1,
            random_state=42,
        )

        return {
            "z_train": z_train,
            "par_train": par_train,
            "y_train": y_train,
            "z_test": z_test,
            "par_test": par_test,
            "y_test": y_test,
        }

    @tf.function
    def train_step(self, input_z=None, input_par=None, output_true=None):
        with tf.GradientTape() as tape:
            loss = self.calc_loss(input_z, input_par, output_true)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": loss,
        }

    def calc_loss(self, input_z=None, input_par=None, output_true=None):
        output_predicted = self.model([input_z, input_par])
        loss = tf.math.reduce_sum(
            tf.math.square(output_predicted - output_true)
        )
        return loss

    def train(
        self,
        learningRateModel,
        batch_size=None,
        nEpochs=None,
        data_file=None,
        gradient_threshold=None,
    ):
        if gradient_threshold is not None:
            print(f"INFO: clipping gradients at {gradient_threshold:.2g}")
        # Make sure the control file for learning rate is consistent with main.py, at least at first
        self.prepareLog()
        bestLoss = None

        if batch_size is None:
            batch_size = self.batch_size
        if nEpochs is None:
            nEpochs = self.nEpochs

        # Make dataset
        if data_file is None:
            data_file = self.np_data_file

        input_dim_data, output_dim_data = self.get_dim_from_data(data_file)
        assert input_dim_data == self.input_dim
        assert output_dim_data == self.output_dim

        dataset = self.make_data(
            scaler_folder=self.model_folder, np_file=data_file
        )
        self.n_batch = dataset["z_train"].shape[0] // batch_size
        self.freq = self.n_batch * 10
        # self.freq = 1

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (
                dataset["z_train"],
                dataset["par_train"],
                dataset["y_train"],
            )
        ).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (
                dataset["z_test"],
                dataset["par_test"],
                dataset["y_test"],
            )
        ).batch(100000)

        # Prepare LR
        lr_m = learningRateModel
        self.optimizer = optimizers.Adam(learning_rate=lr_m)

        # Train
        print_progress_bar(
            0,
            nEpochs,
            prefix="Loss=%s  Epoch= %d / %d " % ("?", 0, nEpochs),
            suffix="Complete",
            length=20,
        )

        for epoch in range(nEpochs):
            train_loss = 0
            time_per_step = 0
            for step, batch in enumerate(train_dataset):
                self.total_step = step + epoch * self.n_batch
                time_s = time.time()
                loss_info = self.train_step(
                    input_z=batch[0], input_par=batch[1], output_true=batch[2]
                )
                loss_val = loss_info["loss"]
                time_e = time.time()
                train_loss = (
                    (step) * train_loss + tf.reduce_sum(loss_val)
                ) / (step + 1)
                time_per_step = (
                    (step) * (time_per_step) + (time_e - time_s)
                ) / (step + 1)
            test_loss = 0
            for val_step, val_batch in enumerate(test_dataset):
                val_loss = self.calc_loss(
                    val_batch[0], val_batch[1], val_batch[2]
                )
                test_loss = (
                    (val_step) * test_loss + tf.reduce_sum(val_loss)
                ) / (val_step + 1)

            print_progress_bar(
                epoch + 1,
                nEpochs,
                prefix=f"train_l={train_loss:.2f}, test_l={test_loss:.2f}, t/step={1e3*time_per_step:.2f} ms,  Epoch= {epoch + 1} / {nEpochs} ",
                suffix="Complete",
                length=20,
            )
            bestLoss = self.logTraining(
                epoch,
                loss=train_loss,
                val_loss=test_loss,
                bestLoss=bestLoss,
            )
            self.model.save_weights(
                os.path.join(self.model_folder, "last.weights.h5"),
                overwrite=True,
            )

    def pred(self, z, par, rescaled=False):
        if len(z.shape) == 1:
            z = np.reshape(z, (-1, 1))
        if len(par.shape) == 1:
            par = np.reshape(par, (-1, 1))
        scaler_z = joblib.load(os.path.join(self.model_folder, "scaler_z.mod"))
        scaler_par = joblib.load(
            os.path.join(self.model_folder, "scaler_par.mod")
        )
        scaler_y = joblib.load(os.path.join(self.model_folder, "scaler_y.mod"))
        z = np.float32(z)
        par = np.float32(par)
        if not rescaled:
            z_in = scaler_z.transform(z)
            par_in = scaler_par.transform(par)
        y_out = self.model([z_in, par_in])
        y = scaler_y.inverse_transform(y_out)
        return y

    def load(self, weight_file):
        self.model.load_weights(weight_file)

    def prepareLog(self):
        os.makedirs(self.model_folder, exist_ok=True)
        os.makedirs(self.log_loss_folder, exist_ok=True)
        try:
            os.remove(os.path.join(self.log_loss_folder, "log.csv"))
        except FileNotFoundError as err:
            print(err)
            pass
        # Make log headers
        f = open(os.path.join(self.log_loss_folder, "log.csv"), "a+")
        f.write("epoch;step;loss;recons_w_loss;recons_chi_loss\n")
        f.close()

    def logTraining(self, epoch, loss, val_loss, bestLoss):
        f = open(os.path.join(self.log_loss_folder, "log.csv"), "a+")
        f.write(
            str(int(epoch))
            + ";"
            + str(int(epoch * self.n_batch))
            + ";"
            + str(loss.numpy())
            + ";"
            + str(val_loss.numpy())
            + ";"
            + "\n"
        )
        f.close()
        if bestLoss is None or loss < bestLoss:
            bestLoss = loss
            self.model.save_weights(
                os.path.join(self.model_folder, "best.weights.h5"),
                overwrite=True,
            )
        return bestLoss

    def get_loss_dat(self):
        lossData = np.genfromtxt(
            os.path.join(self.log_loss_folder, "log.csv"),
            delimiter=";",
            skip_header=1,
        )
        return {
            "epoch": lossData[:, 0],
            "step": lossData[:, 1],
            "train_loss": lossData[:, 2],
            "val_loss": lossData[:, 3],
        }


if __name__ == "__main__":
    myNN = BCR_NN(
        input_dim=4,
        output_dim=1,
        units=[10, 20, 10, 5],
        activation="tanh",
        model_folder="Modeltmp",
        # weight_file='Modeltmp.save/last.weights.h5',
        log_loss_folder="Logtmp",
    )

    myNN.train(
        learningRateModel=1e-3,
        batch_size=32,
        nEpochs=1000,
        data_file="train_data_gh_17_raw.npz",
    )

    from plotsUtil import *

    nz = 32
    nz_true = 32
    A = np.load("train_data_gh_17_raw.npz")
    dat_par = np.repeat(np.reshape(A["x"][0, 1:], (1, -1)), nz, axis=0)
    dat_z = np.reshape(np.linspace(0, 4, nz), (-1, 1))
    pred = myNN.pred(dat_z, dat_par)

    fig = plt.figure()
    plt.plot(pred[:, 0], dat_z[:, 0], "o", label="pred")
    plt.plot(A["y"][:nz_true], A["x"][:nz_true, 0], "x", label="true")
    prettyLabels("gh", "z", 14)
    plotLegend()

    loss_dat = myNN.get_loss_dat()
    fig = plt.figure()
    plt.plot(
        loss_dat["epoch"],
        loss_dat["train_loss"],
        color="k",
        linewidth=3,
        label="train",
    )
    plt.plot(
        loss_dat["epoch"],
        loss_dat["val_loss"],
        color="b",
        linewidth=3,
        label="test",
    )
    prettyLabels("epoch", "loss", 14)
    ax = plt.gca()
    ax.set_yscale("log")
    plotLegend()

    plt.show()
