import random
import warnings

import numpy as np
import optuna
import pandas as pd
from prettyPlot.plotting import *
from scipy.interpolate import RBFInterpolator
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model as tfModel
from tensorflow.keras.models import Sequential

warnings.filterwarnings("ignore")


def check_data_shape(X: np.ndarray, y: np.ndarray) -> None:
    """
    Tune the shape parameter (epsilon) of the multiquadratic RBF

    Parameters
    ----------
    X : np.ndarray
        Design configuration
        Dimension N by d, where N is the number of simulations,
                       and d is the number of design variables
    y: np.ndarray
        Array of quantity of interest
        Dimension N by 1, where N is the number of simulations
    """

    # Same number of samples
    assert X.shape[0] == y.shape[0]
    # Arrays are 2 dimensional
    assert len(X.shape) == 2
    assert len(y.shape) == 2
    # only 1 QoI
    assert y.shape[1] == 1
    print(f"INFO: {X.shape[0]} sim with {X.shape[1]} design variables")


def tune_rbf(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Tune the shape parameter (epsilon) of the multiquadratic RBF

    Parameters
    ----------
    X : np.ndarray
        Design configuration
        Dimension N by d, where N is the number of simulations,
                       and d is the number of design variables
    y: np.ndarray
        Array of quantity of interest
        Dimension N by 1, where N is the number of simulations
    Returns
    ----------
    params: dict
        dictionary of optimal parameters

    """

    check_data_shape(X=X, y=y)

    # Tune the RBFInterpolator with optuna
    # kernel - "multiquadric" (Can try other kernels)
    def objective(trial):
        epsilon = trial.suggest_float("epsilon", 0.1, 10.0, log=False)
        try:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            return np.mean(
                [
                    mean_squared_error(
                        y[test],
                        RBFInterpolator(
                            X[train],
                            y[train],
                            epsilon=epsilon,
                            kernel="multiquadric",
                        )(X[test]),
                    )
                    for train, test in kf.split(X)
                ]
            )
        except:
            return float("inf")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    return study.best_params


def tune_rf(X: np.ndarray, y: np.ndarray):
    """
    Tune the number of trees (n_estimators)
             tree depth (max_depth)
             number of samples in a leaf (min_samples_leaf)
    of the RandomForestRegressor

    Parameters
    ----------
    X : np.ndarray
        Design configuration
        Dimension N by d, where N is the number of simulations,
                       and d is the number of design variables
    y: np.ndarray
        Array of quantity of interest
        Dimension N by 1, where N is the number of simulations

    Returns
    ----------
    params: dict
        dictionary of optimal parameters

    """

    check_data_shape(X=X, y=y)

    # Tune the Random Forest
    def objective(trial):
        model = RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 200),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            random_state=42,
        )
        scores = cross_val_score(
            model, X, y, cv=5, scoring="neg_mean_squared_error"
        )
        return -np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    return study.best_params


def tune_nn(X_encoded: np.ndarray, y: np.ndarray) -> dict:
    """
    Tune the number of neurons (n_units)
             number of layers (n_layers)

    in a leaf (min_samples_leaf) of the RandomForestRegressor

    Parameters
    ----------
    X_encoded : np.ndarray
        Design configuration
        Dimension N by d, where N is the number of simulations,
                       and d is the number of design variables
    y: np.ndarray
        Array of quantity of interest
        Dimension N by 1, where N is the number of simulations

    Returns
    ----------
    params: dict
        dictionary of optimal parameters

    """
    check_data_shape(X=X_encoded, y=y)

    # Tune the Neural Network
    def objective(trial):
        units = trial.suggest_int("n_units", 16, 128)
        layers = trial.suggest_int("n_layers", 1, 3)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        mses = []
        for train, test in kf.split(X_encoded):
            model = Sequential()
            model.add(
                Dense(
                    units, activation="relu", input_shape=(X_encoded.shape[1],)
                )
            )
            for _ in range(layers - 1):
                model.add(Dense(units, activation="relu"))
            model.add(Dense(1))
            model.compile(optimizer="adam", loss="mse")
            model.fit(
                X_encoded[train],
                y[train],
                epochs=100,
                batch_size=16,
                verbose=0,
            )
            mses.append(
                mean_squared_error(
                    y[test], model.predict(X_encoded[test]).flatten()
                )
            )
        return np.mean(mses)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    return study.best_params


class Surrogate_wrapper:
    """
    Wrapper that builds the surrogate model and predicts the QOI value for given X
    """

    def __init__(
        self, model_type: str, X: np.ndarray, y: np.ndarray, params: dict
    ):
        """
        Create the surroagte wrapper

        Parameters
        ----------
        model_type : str
            'rbf' for RBFInterpolator
            'rf' for Random Forest
            'nn' for Neural Network

        X: np.ndarray
            Raw design configuration
            Dimension N by d, where N is the number of simulations,
                           and d is the number of design variables
        y: np.ndarray
            Raw array of quantity of interest
            Dimension N by 1, where N is the number of simulations
        params: dict
            dictionary of surrogate parameters

        """
        self.model_type = model_type
        self.encoder = None

        if model_type.lower() == "rbf":
            self.model = RBFInterpolator(
                X, y, kernel="multiquadric", epsilon=params["epsilon"]
            )
        elif model_type.lower() == "rf":
            self.model = RandomForestRegressor(**params, random_state=42)
            self.model.fit(X, y)
        elif model_type.lower() == "nn":
            self.encoder = OneHotEncoder(sparse_output=False)
            X_encoded = self.encoder.fit_transform(X)
            self.model = self._build_nn(
                X_encoded.shape[1], params["n_units"], params["n_layers"]
            )
            self.model.fit(X_encoded, y, epochs=100, batch_size=16, verbose=0)
        else:
            raise NotImplementedError

    def _build_nn(self, input_dim: int, units: int, layers: int) -> tfModel:
        """
        Builds the neural net

        Parameters
        ----------
        input_dim : int
            number of features
        units: int
            number of neurons per layer
        layers: int
            number of hidden layers

        Returns
        ----------
        model: tfModel
        """
        model = Sequential()
        model.add(Dense(units, activation="relu", input_shape=(input_dim,)))
        for _ in range(layers - 1):
            model.add(Dense(units, activation="relu"))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")
        return model

    def predict(self, X):
        X = X.reshape(1, -1)
        if self.model_type == "nn":
            X = self.encoder.transform(X)
            return float(self.model.predict(X)[0])
        elif self.model_type == "rf":
            return float(self.model.predict(X)[0])
        else:
            return float(self.model(X)[0])


def simulated_annealing_surrogate(
    surrogate: Surrogate_wrapper,
    dim: int = 12,
    max_iters: int = 1000,
    temp: float = 10.0,
    alpha: float = 0.95,
) -> tuple[np.ndarray, float, list[tuple], list[np.ndarray]]:
    """
    Runs the Simulated Annealing (SA) and reports the results

    Parameters
    ----------
    surrogate : Surrogate_wrapper
        tuned surrogate model
    dim: int
        dimension of the problem
    max_iters: int
        maximum number of iterations for SA
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
    trace_x: list[np.ndarray]
        tracks how X changes during each iteration.
    """

    # generate random starting point
    x_curr = np.random.randint(0, 3, size=dim)

    y_curr = surrogate.predict(x_curr)
    x_best, y_best = x_curr.copy(), y_curr

    trace = [(0, y_best)]
    trace_x = [x_curr.copy()]

    for i in range(max_iters):
        # Optimization loop

        x_new = x_curr.copy()
        idx = random.randint(0, dim - 1)
        # perturb a random dimension to get new x
        x_new[idx] = (x_new[idx] + random.choice([-1, 1])) % 3
        y_new = surrogate.predict(x_new)

        delta = y_new - y_curr
        if delta < 0 or np.random.rand() < np.exp(-delta / temp):
            x_curr, y_curr = x_new, y_new
            if y_curr < y_best:
                x_best, y_best = x_new.copy(), y_new

        trace.append((i, y_best))
        trace_x.append(x_curr.copy())
        temp *= alpha
    return x_best, y_best, trace, trace_x


def run_optimization(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "rbf",
    n_runs: int = 10,
    max_iters: int = 1000,
    bootstrap_size: int = 100,
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

    n_runs :int
        number of bootstraps
    max_iters: int
        maximum number of SA iterations
    bootstrap_size : int
        size of bootstrap samples

    Returns
    ----------
    """

    all_x = []
    all_y = []
    all_traces = np.zeros((n_runs, max_iters + 1))
    all_trace_x = []
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
        x_best, y_best, trace, trace_x = simulated_annealing_surrogate(
            surrogate, dim=X.shape[1], max_iters=max_iters
        )

        trace_y = [y for _, y in trace]
        all_traces[i, :] = trace_y
        all_x.append(x_best)
        all_y.append(y_best)
        all_trace_x.append(trace_x)
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
        f"best_bootstrap_solution_{model_type}_size_{bootstrap_size}.csv",
        index=False,
    )

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
        f"Mean_Convergence_plot_{model_type}_size_{bootstrap_size}.png",
        dpi=300,
    )
    plt.savefig(
        f"Mean_Convergence_plot_{model_type}_size_{bootstrap_size}.pdf",
    )
    plt.show()

    # Compute the distance of intermediate x from the optimal solution
    optimal_x = np.ones(X.shape[1], dtype=int)
    l1_traces = []
    for t in all_trace_x:
        distances = [np.sum(np.abs(np.array(x) - optimal_x)) for x in t]
        l1_traces.append(distances)

    """ compute and make the mean-CI plot for the distance between the intermdeiate   
    solution and the optimal solution """
    l1_traces = np.array(l1_traces)
    mean_l1 = np.mean(l1_traces, axis=0)
    std_l1 = np.std(l1_traces, axis=0)
    lower = mean_l1 - 1.96 * std_l1 / np.sqrt(len(trace_x))
    upper = mean_l1 + 1.96 * std_l1 / np.sqrt(len(trace_x))
    iterations = np.arange(len(mean_l1))
    plt.figure(figsize=(8, 4))
    plt.plot(
        iterations,
        mean_l1,
        label="Mean L1 Distance from Optimal",
        color="darkred",
    )
    plt.fill_between(
        iterations, lower, upper, alpha=0.3, color="darkred", label="95% CI"
    )
    pretty_labels(
        "Iteration",
        "L1 Distance from Optimal",
        fontsize=16,
        title=f"Convergence Toward Optimal Solution (L1 Norm)",
        grid=True,
        fontname="Times",
    )
    pretty_legend(fontsize=16, fontname="Times")
    plt.tight_layout()
    plt.savefig(
        f"Mean_L1_distance_{model_type}_size_{bootstrap_size}.png", dpi=300
    )
    plt.savefig(f"Mean_L1_distance_{model_type}_size_{bootstrap_size}.pdf")
    plt.show()


if __name__ == "__main__":

    # Read data from the csv file.
    X_raw_data = pd.read_csv("Xdata_study_scaleup_0_4vvm_3000W.csv")
    y_raw_data = pd.read_csv("ydata_study_scaleup_0_4vvm_3000W.csv")

    X = X_raw_data.values
    y = y_raw_data.iloc[:, :-1].values
    # We want to maximize, so we minimize the opposite
    y = y * -1

    # The function will build, tune, and optimize the surrogate model and postprocess the results.
    run_optimization(
        X, y, model_type="rbf", n_runs=10, max_iters=1000, bootstrap_size=150
    )
