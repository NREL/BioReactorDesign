import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from surrogate import Surrogate_wrapper, tune_nn, tune_rbf, tune_rf


def make_or_load_split(
    study_path: str,
    n_samples: int,
    n_test: int = 20,
    n_val: int = 20,
    seed: int = 0,
) -> dict:
    """Return {'train','test','val'} row-index arrays, frozen in split.csv.

    If split.csv already exists in ``study_path`` it is loaded verbatim so the
    split is reproducible across runs; otherwise it is drawn deterministically
    from ``seed`` and written.
    """
    split_file = os.path.join(study_path, "split.csv")
    if os.path.exists(split_file):
        df = pd.read_csv(split_file)
        return {
            name: df.loc[df["set"] == name, "row_index"].to_numpy()
            for name in ("train", "test", "val")
        }

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    test_idx = perm[:n_test]
    val_idx = perm[n_test : n_test + n_val]
    train_idx = perm[n_test + n_val :]

    rows = (
        [(int(i), "test") for i in test_idx]
        + [(int(i), "val") for i in val_idx]
        + [(int(i), "train") for i in train_idx]
    )
    pd.DataFrame(rows, columns=["row_index", "set"]).to_csv(
        split_file, index=False
    )
    return {"train": train_idx, "test": test_idx, "val": val_idx}


def _predict_batch(surrogate: Surrogate_wrapper, X: np.ndarray) -> np.ndarray:
    """Row-wise predictions (Surrogate_wrapper.predict is single-sample)."""
    return np.array([surrogate.predict(X[i]) for i in range(len(X))])


def select_or_load_surrogate(
    X: np.ndarray,
    y: np.ndarray,
    split: dict,
    qoi_name: str,
    out_folder: str,
    n_trials: int = 100,
) -> tuple[str, dict]:
    """Tune RBF, RF, and NN on the test set, pick the winner on the validation set.

    Caches the full result to ``hyperparams_{qoi_name}.json`` and reloads it on
    subsequent calls so both drivers reuse one selection. Returns
    ``(best_model_type, best_params)``.
    """
    os.makedirs(out_folder, exist_ok=True)
    json_file = os.path.join(out_folder, f"hyperparams_{qoi_name}.json")
    if os.path.exists(json_file):
        with open(json_file) as f:
            record = json.load(f)
        return record["best"], record["best_params"]

    X_train, y_train = X[split["train"]], y[split["train"]]
    X_test, y_test = X[split["test"]], y[split["test"]]
    X_val, y_val = X[split["val"]], y[split["val"]]

    # physical QoI reference values (y is the negated min-space target, so
    # physical QoI = -y). Amplitude = max - min = the CFD data range height.
    physical = -y
    amp_physical = float(np.max(physical) - np.min(physical))
    mean_physical = float(np.mean(physical))

    def nrmse_pct(mse: float, denom: float) -> float:
        """Normalized RMSE as a percent of a physical-QoI reference."""
        if denom == 0:
            return float("nan")
        return 100.0 * np.sqrt(mse) / abs(denom)

    # 1. HPO scored on the test set
    tuners = {"rbf": tune_rbf, "rf": tune_rf, "nn": tune_nn}
    tuned = {
        name: tuner(X_train, y_train, X_test, y_test, n_trials=n_trials)
        for name, tuner in tuners.items()
    }

    # 2. compare on the validation set (each refit on the train pool). MSE is in
    # physical units since Surrogate_wrapper.predict inverse-transforms.
    results = {}
    for model_type, params in tuned.items():
        surrogate = Surrogate_wrapper(model_type, X_train, y_train, params)
        test_mse = mean_squared_error(
            y_test.ravel(), _predict_batch(surrogate, X_test)
        )
        val_mse = mean_squared_error(
            y_val.ravel(), _predict_batch(surrogate, X_val)
        )
        results[model_type] = {
            "params": params,
            "test_mse": float(test_mse),
            "val_mse": float(val_mse),
            "test_nrmse_amp_pct": float(nrmse_pct(test_mse, amp_physical)),
            "test_nrmse_mean_pct": float(nrmse_pct(test_mse, mean_physical)),
            "val_nrmse_amp_pct": float(nrmse_pct(val_mse, amp_physical)),
            "val_nrmse_mean_pct": float(nrmse_pct(val_mse, mean_physical)),
        }

    best = min(results, key=lambda m: results[m]["val_mse"])
    record = {
        "qoi": qoi_name,
        "best": best,
        "best_params": results[best]["params"],
        "amp_physical": amp_physical,
        "mean_physical": mean_physical,
        **results,
    }
    with open(json_file, "w") as f:
        json.dump(record, f, indent=2)

    val_summary = ", ".join(
        f"{m.upper()} val_mse={results[m]['val_mse']:.6g}"
        f" (nRMSE amp={results[m]['val_nrmse_amp_pct']:.1f}%"
        f" mean={results[m]['val_nrmse_mean_pct']:.1f}%)"
        for m in results
    )
    print(f"  {qoi_name}: {val_summary} -> best={best}")
    for m in results:
        print(f"    {m.upper()} params: {results[m]['params']}")
    return best, results[best]["params"]
