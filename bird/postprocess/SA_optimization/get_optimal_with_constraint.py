import os
import random

import numpy as np
import optuna
import pandas as pd
from prettyPlot.plotting import *
from sklearn.preprocessing import OneHotEncoder

from bird.postprocess.SA_optimization.surrogate import (
    Surrogate_wrapper,
    tune_nn,
    tune_rbf,
    tune_rf,
)


def simulated_annealing_surrogate(
    surrogate: Surrogate_wrapper,
    dim: int = 12,
    max_iters: int = 1000,
    max_spargers: int = 8,
    temp: float = 10.0,
    alpha: float = 0.95,
) -> tuple[np.ndarray, float, list[tuple]]:
    """
    Runs the Simulated Annealing (SA) constrained by the number of spargers and reports the results

    Parameters
    ----------
    surrogate : Surrogate_wrapper
        tuned surrogate model
    dim: int
        dimension of the problem
    max_iters: int
        maximum number of iterations for SA
    max_spargers: int
        maximum number of spargers (this is a constraint of the optimization problem)
    temp: float
        parameter of Simulated Annealing (can be changed or tuned for other problems)
        max temperature. It controls the exploration rate of SA
    alpha:
        parameter of Simulated Annealing (can be changed or tuned for other problems)
        cooling rate. It determines how quickly temp decays

    Returns
    ----------
    x_best: np.ndarray
        optimal solution
    y_best: float
        optimal objective function value
    trace: list[tuple]
        optimization log. Updates at the end of each iteration.
    """

    def is_valid(x):
        # Checks if the number of spargers <= max_spargers
        return np.sum(x == 1) <= max_spargers

    while True:
        # generate random starting point
        x_curr = np.random.randint(0, 3, size=dim)
        if is_valid(x_curr):
            break

    y_curr = surrogate.predict(x_curr)
    x_best, y_best = x_curr.copy(), y_curr

    trace = [(0, y_best)]

    for i in range(max_iters):
        # Optimization loop

        x_new = x_curr.copy()
        idx = random.randint(0, dim - 1)
        # perturb a random dimension to get new x
        x_new[idx] = (x_new[idx] + random.choice([-1, 1])) % 3

        if not is_valid(x_new):
            trace.append((i, y_best))
            temp *= alpha
            continue

        y_new = surrogate.predict(x_new)

        delta = y_new - y_curr
        if delta < 0 or np.random.rand() < np.exp(-delta / temp):
            x_curr, y_curr = x_new, y_new
            if y_curr < y_best:
                x_best, y_best = x_new.copy(), y_new

        trace.append((i, y_best))
        temp *= alpha

    return x_best, y_best, trace


def run_optimization(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "rbf",
    max_spargers: int = 8,
    n_runs: int = 10,
    max_iters: int = 1000,
    bootstrap_size: int = 100,
    out_folder: str = ".",
):
    """
    Bootstraps data, runs optimization and postprocesses the results.

    Parameters
    ----------
    X: np.ndarray
        Raw design configuration
        Dimension N by d, where N is the number of simulations,
                       and d is the number of design variables
    y: np.ndarray
        Raw array of quantity of interest
        Dimension N by 1, where N is the number of simulations

    model_type : str
        'rbf' for RBFInterpolator
        'rf' for Random Forest
        'nn' for Neural Network

    max_spargers: int
        maximum number of spargers (this is a constraint of the optimization problem)
    n_runs :int
        number of bootstraps
    max_iters: int
        maximum number of SA iterations
    bootstrap_size : int
        size of bootstrap samples
    out_folder: str
        folder where to output the results

    Returns
    ----------
    """

    all_x = []
    all_y = []
    all_traces = np.zeros((n_runs, max_iters + 1))
    rng = np.random.default_rng(42)
    # Bootstrap the data
    bootstrap_idxs = [
        rng.choice(len(X), size=bootstrap_size, replace=False)
        for _ in range(n_runs)
    ]

    i = 0
    for idxs in bootstrap_idxs:
        # Tune the model for each bootstrap
        X_sub, y_sub = X[idxs], y[idxs]
        if model_type == "rbf":
            params = tune_rbf(X_sub, y_sub)
        elif model_type == "rf":
            params = tune_rf(X_sub, y_sub)
        elif model_type == "nn":
            encoder = OneHotEncoder(sparse_output=False)
            X_encoded = encoder.fit_transform(X_sub)
            params = tune_nn(X_encoded, y_sub)
        else:
            raise ValueError("Invalid model_type")

        # build the surrogate model
        surrogate = Surrogate_wrapper(model_type, X_sub, y_sub, params)

        # run optimization
        x_best, y_best, trace = simulated_annealing_surrogate(
            surrogate,
            dim=X.shape[1],
            max_iters=max_iters,
            max_spargers=max_spargers,
        )

        trace_y = [y for _, y in trace]
        all_traces[i, :] = trace_y
        all_x.append(x_best)
        all_y.append(y_best)
        i = i + 1

    # Compute the best y across all the bootstraps
    best_index = np.argmin(all_y)
    best_x = all_x[best_index]
    best_y = all_y[best_index]
    # Save the best solution
    df = pd.DataFrame(
        [
            {
                **{f"x{i}": best_x[i] for i in range(len(best_x))},
                "best_y": best_y,
            }
        ]
    )
    df.to_csv(
        os.path.join(
            out_folder,
            f"best_bootstrap_solution_{model_type}_size_{bootstrap_size}_max_spargers_{max_spargers}.csv",
        ),
        index=False,
    )

    print("X = ", x_best)
    print("surrogate-predicted y", y_best)

    # Make the mean-CI plot for the objective function and save it
    mean_trace = np.mean(-1 * all_traces, axis=0)
    std_trace = np.std(all_traces, axis=0)
    lower_bound = mean_trace - 1.96 * std_trace / np.sqrt(n_runs)
    upper_bound = mean_trace + 1.96 * std_trace / np.sqrt(n_runs)
    iterations = np.arange(max_iters + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(
        iterations,
        mean_trace,
        label=f"Mean Convergence {model_type.upper()}",
        color="blue",
    )
    plt.fill_between(
        iterations,
        lower_bound,
        upper_bound,
        color="blue",
        alpha=0.3,
        label="95% CI",
    )
    pretty_labels(
        "Iteration",
        "Best Surrogate-Predicted Objective",
        fontsize=16,
        title=f"Mean Convergence with 95% Confidence Interval ({model_type.upper()})",
        grid=True,
        fontname="Times",
    )
    pretty_legend(fontsize=16, fontname="Times")
    plt.savefig(
        os.path.join(
            out_folder,
            f"Mean_Convergence_plot_{model_type}_size_{bootstrap_size}_max_spargers_{max_spargers}.png",
        ),
        dpi=300,
    )
    plt.savefig(
        os.path.join(
            out_folder,
            f"Mean_Convergence_plot_{model_type}_size_{bootstrap_size}_max_spargers_{max_spargers}.pdf",
        ),
    )
    plt.show()


if __name__ == "__main__":
    studies = ["study_scaleup_0_4vvm_3000W"]

    for study in studies:
        for nsparg in [1, 2, 3, 4, 5, 6, 7, 8]:
            X_raw_data = pd.read_csv(os.path.join(study, "Xdata.csv"))
            y_raw_data = pd.read_csv(os.path.join(study, "ydata.csv"))

            X = X_raw_data.values
            y = y_raw_data.iloc[:, :-1].values
            # We want to maximize, so we minimize the opposite
            y = y * -1

            # The function will build, tune, and optimize the surrogate model and postprocess the results.
            run_optimization(
                X,
                y,
                model_type="rbf",
                max_spargers=nsparg,
                n_runs=10,
                max_iters=1000,
                bootstrap_size=150,
                out_folder=study,
            )
