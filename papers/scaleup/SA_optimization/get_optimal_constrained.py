import csv
import os
import random
import warnings

import numpy as np
from load_data import load_full_dataset
from model_selection import make_or_load_split, select_or_load_surrogate
from surrogate import Surrogate_wrapper

warnings.filterwarnings("ignore")

QOI_COLS = {
    "qoi": {"label": "QOI"},
    "qoi_kla": {"label": "QOI Num"},
    "qoi_sum": {"label": "QOI Sum"},
    "qoi_sum_kla": {"label": "QOI Sum Num"},
}


def simulated_annealing_constrained(
    surrogate: Surrogate_wrapper,
    dim: int = 11,
    max_iters: int = 1000,
    max_spargers: int = 8,
    temp: float = 10.0,
    alpha: float = 0.95,
) -> tuple[np.ndarray, float, list[tuple]]:
    """SA constrained by max number of spargers."""

    def is_valid(x):
        return np.sum(x == 1) <= max_spargers

    while True:
        x_curr = np.random.randint(0, 3, size=dim)
        if is_valid(x_curr):
            break

    y_curr = surrogate.predict(x_curr)
    x_best, y_best = x_curr.copy(), y_curr
    trace = [(0, y_best)]

    for i in range(max_iters):
        x_new = x_curr.copy()
        idx = random.randint(0, dim - 1)
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


def run_constrained_optimization(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    params: dict,
    max_spargers: int = 8,
    n_runs: int = 10,
    max_iters: int = 1000,
    bootstrap_size: int = 250,
) -> tuple[float, float]:
    """Mean optimal QoI over the bootstrap optima and its 95% uncertainty.

    Each bootstrap subset refits the fixed, pre-tuned surrogate
    (``model_type`` + ``params``) and yields one constrained optimum. Returns
    the mean of those optima (positive, i.e. negated back to physical) and the
    95% confidence interval of that mean (1.96 * std / sqrt(n_runs)).
    """
    rng = np.random.default_rng(42)
    bs = min(bootstrap_size, len(X))
    bootstrap_idxs = [
        rng.choice(len(X), size=bs, replace=False) for _ in range(n_runs)
    ]

    all_y = []
    for idxs in bootstrap_idxs:
        X_sub, y_sub = X[idxs], y[idxs]
        surrogate = Surrogate_wrapper(model_type, X_sub, y_sub, params)
        _, y_best, _ = simulated_annealing_constrained(
            surrogate,
            dim=X.shape[1],
            max_iters=max_iters,
            max_spargers=max_spargers,
        )
        all_y.append(y_best)

    optima = -np.array(all_y)  # negate back to physical QoI
    mean_optimum = float(np.mean(optima))
    ci95 = float(1.96 * np.std(optima) / np.sqrt(len(optima)))
    return mean_optimum, ci95


if __name__ == "__main__":
    studies = {
        "data/study/study_0_4vvm_lev1": "lev1",
        "data/study/study_0_4vvm_lev6": "lev6",
    }

    results = []

    for study_path, level_label in studies.items():
        print(f"\n=== {level_label}: {study_path} ===")
        X, Y_full = load_full_dataset(study_path)

        split = make_or_load_split(
            study_path, n_samples=len(X), n_test=20, n_val=20
        )

        for qoi_name, qoi_info in QOI_COLS.items():
            y = Y_full[qoi_name].values.reshape(-1, 1) * -1

            # winning surrogate + tuned params (cached from the selection step)
            out = os.path.join(study_path, qoi_name)
            model_type, params = select_or_load_surrogate(
                X, y, split, qoi_name, out
            )

            for nsparg in range(3, 10):
                print(
                    f"  {qoi_name} ({model_type}), max_spargers={nsparg}...",
                    end="",
                    flush=True,
                )
                mean_optimum, ci95 = run_constrained_optimization(
                    X,
                    y,
                    model_type=model_type,
                    params=params,
                    max_spargers=nsparg,
                    n_runs=10,
                    max_iters=1000,
                    bootstrap_size=250,
                )
                print(f" mean={mean_optimum:.6g} ci95={ci95:.6g}")
                results.append(
                    {
                        "level": level_label,
                        "qoi_name": qoi_name,
                        "model_type": model_type,
                        "n_spargers": nsparg,
                        "optimal_qoi_mean": mean_optimum,
                        "optimal_qoi_ci95": ci95,
                    }
                )

    out_file = "marginal_gain_results.csv"
    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "level",
                "qoi_name",
                "model_type",
                "n_spargers",
                "optimal_qoi_mean",
                "optimal_qoi_ci95",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote {out_file}")
