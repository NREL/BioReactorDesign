import random
import warnings

import numpy as np
import optuna
import pandas as pd
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
