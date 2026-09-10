import warnings

import numpy as np
import optuna
import torch
import torch.nn as nn
from scipy.interpolate import RBFInterpolator
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# epsilon-using RBF kernels (so tuning epsilon stays meaningful for each)
RBF_KERNELS = (
    "multiquadric",
    "inverse_multiquadric",
    "inverse_quadratic",
    "gaussian",
)

# the 11 design variables are unordered categoricals {0=mixer, 1=sparger, 2=wall}
DESIGN_CATEGORIES = (0, 1, 2)


def make_encoder(n_features: int = 11) -> OneHotEncoder:
    """One-hot encoder shared by all surrogates.

    Categories are fixed to {0, 1, 2} for every design variable so the encoded
    width is always 3 x n_features, regardless of which values happen to appear
    in a given train/bootstrap subset.
    """
    return OneHotEncoder(
        categories=[list(DESIGN_CATEGORIES)] * n_features,
        sparse_output=False,
        handle_unknown="ignore",
    )


def check_data_shape(X: np.ndarray, y: np.ndarray) -> None:
    """Validate shapes: (N, d) for X, (N, 1) for y."""
    assert X.shape[0] == y.shape[0]
    assert len(X.shape) == 2
    assert len(y.shape) == 2
    assert y.shape[1] == 1
    print(f"INFO: {X.shape[0]} sim with {X.shape[1]} design variables")


def tune_rbf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_trials: int = 100,
) -> dict:
    """Tune RBF epsilon + kernel type by MSE on a held-out test set."""
    check_data_shape(X=X_train, y=y_train)
    check_data_shape(X=X_test, y=y_test)
    encoder = make_encoder(X_train.shape[1])
    X_train_enc = encoder.fit_transform(X_train)
    X_test_enc = encoder.transform(X_test)
    y_scaler = StandardScaler().fit(y_train)
    y_train_s = y_scaler.transform(y_train)
    y_test_s = y_scaler.transform(y_test)

    def objective(trial):
        kernel = trial.suggest_categorical("kernel", list(RBF_KERNELS))
        epsilon = trial.suggest_float("epsilon", 0.1, 10.0, log=False)
        try:
            model = RBFInterpolator(
                X_train_enc, y_train_s, epsilon=epsilon, kernel=kernel
            )
            return mean_squared_error(y_test_s, model(X_test_enc))
        except Exception:
            return float("inf")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def tune_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_trials: int = 100,
) -> dict:
    """Tune RandomForest hyperparameters by MSE on a held-out test set."""
    check_data_shape(X=X_train, y=y_train)
    check_data_shape(X=X_test, y=y_test)
    encoder = make_encoder(X_train.shape[1])
    X_train_enc = encoder.fit_transform(X_train)
    X_test_enc = encoder.transform(X_test)
    y_scaler = StandardScaler().fit(y_train)
    y_train_s = y_scaler.transform(y_train)
    y_test_s = y_scaler.transform(y_test)

    def objective(trial):
        model = RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 200),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_float("max_features", 0.1, 1.0),
            random_state=42,
        )
        model.fit(X_train_enc, y_train_s.ravel())
        return mean_squared_error(y_test_s, model.predict(X_test_enc))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def tune_nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_trials: int = 100,
) -> dict:
    """Tune NN units, layers, and learning rate by MSE on a held-out test set."""
    check_data_shape(X=X_train, y=y_train)
    check_data_shape(X=X_test, y=y_test)
    encoder = make_encoder(X_train.shape[1])
    X_train_enc = encoder.fit_transform(X_train)
    X_test_enc = encoder.transform(X_test)
    y_scaler = StandardScaler().fit(y_train)
    y_train_s = y_scaler.transform(y_train)
    y_test_s = y_scaler.transform(y_test)

    def objective(trial):
        units = trial.suggest_int("n_units", 16, 128)
        layers = trial.suggest_int("n_layers", 1, 3)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        model = _build_nn(X_train_enc.shape[1], units, layers)
        _train_nn(model, X_train_enc, y_train_s, lr=lr)
        with torch.no_grad():
            X_t = torch.tensor(X_test_enc, dtype=torch.float32)
            preds = model(X_t).numpy().flatten()
        return mean_squared_error(y_test_s, preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def _build_nn(input_dim: int, units: int, layers: int) -> nn.Sequential:
    """Build a PyTorch Sequential model."""
    layer_list = [nn.Linear(input_dim, units), nn.ReLU()]
    for _ in range(layers - 1):
        layer_list.extend([nn.Linear(units, units), nn.ReLU()])
    layer_list.append(nn.Linear(units, 1))
    return nn.Sequential(*layer_list)


def _train_nn(
    model: nn.Sequential,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 1000,
    lr: float = 1e-3,
    patience: int = 50,
    val_frac: float = 0.15,
    min_delta: float = 1e-5,
) -> None:
    """Train in-place (full-batch Adam) with early stopping.

    A fraction ``val_frac`` of the passed data is held out to monitor validation
    loss; training stops once it has not improved by ``min_delta`` for
    ``patience`` consecutive epochs, and the best weights are restored. The
    holdout is carved from the training data given here, so it never touches the
    outer test/validation split.
    """
    n_samples = X.shape[0]
    n_val = int(round(n_samples * val_frac))
    perm = np.random.default_rng(0).permutation(n_samples)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    if n_val < 1 or train_idx.size < 1:  # too small to hold out: monitor train
        val_idx = train_idx = np.arange(n_samples)

    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.float32)
    X_val = torch.tensor(X[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y[val_idx], dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss_fn(model(X_train), y_train).backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(X_val), y_val))
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state = {
                name: t.detach().clone()
                for name, t in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()


class Surrogate_wrapper:
    """Unified predict() interface for RBF, RF, and NN surrogates.

    All three share one one-hot encoding of the raw {0, 1, 2} design variables
    and one standardization of the targets, both fit here on the training data.
    Targets are standardized before fitting; predict() inverse-transforms so its
    return value is always in physical units. predict() takes a raw design vector
    and encodes it, so the caller (simulated annealing) stays in raw design space.
    """

    def __init__(
        self,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        params: dict,
    ):
        self.model_type = model_type.lower()
        self.encoder = make_encoder(X.shape[1])
        X_encoded = self.encoder.fit_transform(X)
        self.y_scaler = StandardScaler().fit(y)
        y_scaled = self.y_scaler.transform(y)

        if self.model_type == "rbf":
            self.model = RBFInterpolator(
                X_encoded,
                y_scaled,
                kernel=params.get("kernel", "multiquadric"),
                epsilon=params["epsilon"],
            )
        elif self.model_type == "rf":
            self.model = RandomForestRegressor(**params, random_state=42)
            self.model.fit(X_encoded, y_scaled.ravel())
        elif self.model_type == "nn":
            self.model = _build_nn(
                X_encoded.shape[1],
                params["n_units"],
                params["n_layers"],
            )
            _train_nn(
                self.model, X_encoded, y_scaled, lr=params.get("lr", 1e-3)
            )
        else:
            raise NotImplementedError(f"Unknown model_type: {model_type}")

    def predict(self, X: np.ndarray) -> float:
        """Return the physical-space prediction for one raw design vector."""
        X_enc = self.encoder.transform(X.reshape(1, -1))
        if self.model_type == "nn":
            with torch.no_grad():
                X_t = torch.tensor(X_enc, dtype=torch.float32)
                scaled = self.model(X_t).item()
        elif self.model_type == "rf":
            scaled = self.model.predict(X_enc)[0]
        else:
            scaled = float(np.squeeze(self.model(X_enc)))
        return float(self.y_scaler.inverse_transform([[scaled]])[0, 0])
