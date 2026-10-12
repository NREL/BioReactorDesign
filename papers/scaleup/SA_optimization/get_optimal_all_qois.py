import argparse
import csv
import json
import os
import random
import warnings

import numpy as np
import pandas as pd
from load_data import load_full_dataset
from matplotlib.patches import Patch
from model_selection import make_or_load_split, select_or_load_surrogate
from plot_design_schematic import plot_schematic
from prettyPlot.plotting import plt, pretty_labels, pretty_legend
from surrogate import Surrogate_wrapper

warnings.filterwarnings("ignore")

QOI_COLS = {
    "qoi": {
        "label": r"QOI$_1$",
        "ylabel": r"Optimal QOI$_1$ [kg$^2$/kWh$^2$]",
    },
    "qoi_kla": {
        "label": r"QOI$_2$",
        "ylabel": r"Optimal QOI$_2$ [kg$^2$/s$^2$]",
    },
    "qoi_sum": {
        "label": r"QOI$_3$",
        "ylabel": r"Optimal QOI$_3$ [kg/kWh]",
    },
    "qoi_sum_kla": {
        "label": r"QOI$_4$",
        "ylabel": r"Optimal QOI$_4$ [kg/s]",
    },
}


def simulated_annealing_surrogate(
    surrogate: Surrogate_wrapper,
    dim: int = 11,
    max_iters: int = 1000,
    temp: float = 10.0,
    alpha: float = 0.95,
) -> tuple[np.ndarray, float, list[tuple], list[np.ndarray]]:
    """Run SA on the surrogate and return optimal x, y, traces."""
    x_curr = np.random.randint(0, 3, size=dim)
    y_curr = surrogate.predict(x_curr)
    x_best, y_best = x_curr.copy(), y_curr

    trace = [(0, y_best)]
    trace_x = [x_curr.copy()]

    for i in range(max_iters):
        x_new = x_curr.copy()
        idx = random.randint(0, dim - 1)
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
    model_type: str,
    params: dict,
    n_runs: int = 10,
    max_iters: int = 1000,
    bootstrap_size: int = 250,
    out_folder: str = ".",
    qoi_name: str = "qoi",
    qoi_ylabel: str = "QOI",
) -> None:
    """Bootstrap + SA optimization for one QoI on a fixed, pre-tuned surrogate.

    Hyperparameters are held fixed (``params``); each of ``n_runs`` bootstrap
    subsets (size ``bootstrap_size``, drawn without replacement from the full
    data) only refits the surrogate on its subset.
    """
    os.makedirs(out_folder, exist_ok=True)

    all_x = []
    all_y = []
    all_traces = np.zeros((n_runs, max_iters + 1))
    rng = np.random.default_rng(42)
    bs = min(bootstrap_size, len(X))
    bootstrap_idxs = [
        rng.choice(len(X), size=bs, replace=False) for _ in range(n_runs)
    ]

    for i, idxs in enumerate(bootstrap_idxs):
        X_sub, y_sub = X[idxs], y[idxs]
        surrogate = Surrogate_wrapper(model_type, X_sub, y_sub, params)
        x_best, y_best, trace, trace_x = simulated_annealing_surrogate(
            surrogate, dim=X.shape[1], max_iters=max_iters
        )

        trace_y = [t[1] for t in trace]
        all_traces[i, :] = trace_y
        all_x.append(x_best)
        all_y.append(y_best)

    best_index = np.argmin(all_y)
    best_x = all_x[best_index]
    best_y = all_y[best_index]

    df = pd.DataFrame(
        [
            {
                **{f"x{j}": best_x[j] for j in range(len(best_x))},
                "best_y": best_y,
            }
        ]
    )
    df.to_csv(
        os.path.join(
            out_folder,
            f"best_bootstrap_solution_{model_type}_{qoi_name}.csv",
        ),
        index=False,
    )
    print(f"  {qoi_name} ({model_type}): best_y = {-best_y:.6g}, x = {best_x}")

    # schematic of the optimum design on the reactor bottom face
    plot_schematic(
        best_x,
        os.path.join(out_folder, f"Optimum_schematic_{model_type}_{qoi_name}"),
        title=f"Optimum {qoi_name} ({model_type.upper()})",
    )

    # save the convergence-plot inputs so the plot can be regenerated later
    # (via plot_convergence / --plot-only) without rerunning the optimization
    np.savez(
        _convergence_data_path(out_folder, model_type, qoi_name),
        all_traces=all_traces,
        qoi_lo=float((-y).min()),
        qoi_hi=float((-y).max()),
        best_x=np.asarray(best_x),
        best_y=best_y,
        all_x=np.asarray(all_x),
        all_y=np.asarray(all_y),
    )
    plot_convergence(out_folder, model_type, qoi_name, qoi_ylabel)


def _convergence_data_path(
    out_folder: str, model_type: str, qoi_name: str
) -> str:
    """Path of the saved convergence-plot data for one model/QoI."""
    return os.path.join(
        out_folder, f"convergence_data_{model_type}_{qoi_name}.npz"
    )


def plot_convergence(
    out_folder: str,
    model_type: str,
    qoi_name: str,
    qoi_ylabel: str,
) -> None:
    """Draw the convergence plot from data saved by run_optimization.

    Reads convergence_data_{model_type}_{qoi_name}.npz (the per-run SA traces
    plus the CFD QoI range) so the plot can be regenerated without rerunning the
    optimization. The grey band + dashed lines mark the CFD QoI range so
    extrapolation of the optimum is easy to spot.
    """
    data = np.load(_convergence_data_path(out_folder, model_type, qoi_name))
    all_traces = data["all_traces"]
    qoi_lo = float(data["qoi_lo"])
    qoi_hi = float(data["qoi_hi"])
    n_runs, n_points = all_traces.shape
    iterations = np.arange(n_points)

    mean_trace = np.mean(-1 * all_traces, axis=0)
    std_trace = np.std(all_traces, axis=0)
    lower_bound = mean_trace - 1.96 * std_trace / np.sqrt(n_runs)
    upper_bound = mean_trace + 1.96 * std_trace / np.sqrt(n_runs)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axhspan(
        qoi_lo,
        qoi_hi,
        color="grey",
        alpha=0.25,
        zorder=0,
        label="CFD data range",
    )
    ax.axhline(qoi_lo, color="dimgray", ls="--", lw=1.2, zorder=5)
    ax.axhline(qoi_hi, color="dimgray", ls="--", lw=1.2, zorder=5)
    ax.plot(
        iterations,
        mean_trace,
        label=f"Mean Convergence {model_type.upper()}",
        color="blue",
    )
    ax.fill_between(
        iterations,
        lower_bound,
        upper_bound,
        color="blue",
        alpha=0.3,
        label="95% CI",
    )
    # optional red dashed line: CFD-verified QoI of an optimal design, if provided
    cfd_file = os.path.join(out_folder, "cfd_all_spargers.txt")
    if os.path.exists(cfd_file):
        with open(cfd_file) as f:
            cfd_value = float(f.read().strip().split(",")[0])
        ax.axhline(
            cfd_value,
            color="red",
            ls="--",
            lw=3,
            zorder=6,
            label="All spargers",
        )
    pretty_labels(
        "Iteration",
        qoi_ylabel,
        fontsize=30,
        # title=f"Convergence ({model_type.upper()}, {qoi_ylabel})",
        grid=False,
        fontname="Times",
    )
    pretty_legend(fontsize=30, fontname="Times")
    fname = f"Mean_Convergence_plot_{model_type}_{qoi_name}"
    plt.savefig(os.path.join(out_folder, f"{fname}.png"), dpi=300)
    plt.savefig(os.path.join(out_folder, f"{fname}.pdf"))
    plt.close()


# design choices shown in the bar plot: (code, bar color, name). Only spargers
# (code 1) are shown; mixers (0) and walls (2) are omitted as less
# physically meaningful. Order sets the bar order per QoI.
DESIGN_CHOICES = [(1, "tab:blue", "Sparger"), (0, "tab:red", "Mixers")]


def plot_design_composition(studies: dict, out_path: str) -> None:
    """Bar plot of the mean sparger/mixer/wall count over the bootstrap optima.

    For each (level, QoI) it reads the saved ensemble (all_x) of the winning
    surrogate and, per placement type, plots the mean count across the bootstrap
    optima with an error bar for the std. Blue = sparger, red = mixer,
    grey = wall; lev1 hatched, lev6 solid. Reads only saved data, so it needs no
    optimization rerun.
    """
    qoi_names = list(QOI_COLS.keys())
    # stats[level][qoi_name][choice] = (mean_count, std_count)
    stats = {label: {} for label in studies.values()}
    for study_path, label in studies.items():
        for qoi_name in qoi_names:
            out = os.path.join(study_path, qoi_name)
            json_path = os.path.join(out, f"hyperparams_{qoi_name}.json")
            if not os.path.exists(json_path):
                continue
            with open(json_path) as f:
                best_type = json.load(f)["best"]
            data_path = _convergence_data_path(out, best_type, qoi_name)
            if not os.path.exists(data_path):
                continue
            npz = np.load(data_path)
            if "all_x" not in npz.files:
                print(
                    f"  {label}/{qoi_name}: npz has no all_x (rerun to save)"
                )
                continue
            all_x = npz["all_x"]
            stats[label][qoi_name] = {
                choice: (
                    float((all_x == choice).sum(axis=1).mean()),
                    float((all_x == choice).sum(axis=1).std()),
                )
                for choice, _, _ in DESIGN_CHOICES
            }

    ordered_levels = sorted(studies.values())  # lev1 (hatched) before lev6
    n_bars = len(DESIGN_CHOICES) * len(ordered_levels)
    bar_width = 0.8 / n_bars
    x_ticks = np.arange(len(qoi_names))

    fig, ax = plt.subplots(figsize=(11, 6))
    slot = 0
    for choice, color, _ in DESIGN_CHOICES:
        for level in ordered_levels:
            means = [
                stats[level].get(q, {}).get(choice, (np.nan, np.nan))[0]
                for q in qoi_names
            ]
            stds = [
                stats[level].get(q, {}).get(choice, (np.nan, np.nan))[1]
                for q in qoi_names
            ]
            offset = (slot - (n_bars - 1) / 2) * bar_width
            ax.bar(
                x_ticks + offset,
                means,
                bar_width,
                yerr=stds,
                capsize=3,
                color=color,
                edgecolor="black",
                hatch="//" if level == "lev1" else None,
            )
            slot += 1

    pretty_labels(
        "",
        "Average count over bootstrap optima",
        fontsize=20,
        grid=True,
        fontname="Times",
    )
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([QOI_COLS[q]["label"] for q in qoi_names])
    legend_handles = [
        Patch(facecolor=color, edgecolor="black", label=name)
        for _, color, name in DESIGN_CHOICES
    ] + [
        Patch(facecolor="white", edgecolor="black", hatch="//", label="3.6L"),
        Patch(facecolor="white", edgecolor="black", label=r"608 m$^3$"),
    ]
    ax.legend(handles=legend_handles, fontsize=14)
    plt.savefig(f"{out_path}.png", dpi=300)
    plt.savefig(f"{out_path}.pdf")
    plt.close()


if __name__ == "__main__":
    studies = {
        "data/study/study_0_4vvm_lev6": "lev6",
        "data/study/study_0_4vvm_lev1": "lev1",
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="skip the optimization and regenerate the convergence plots from "
        "the saved convergence_data_*.npz in each qoi folder",
    )
    args = parser.parse_args()

    selection_rows = []
    for study_path, label in studies.items():
        print(f"\n=== {label}: {study_path} ===")
        if not args.plot_only:
            X, Y_full = load_full_dataset(study_path)
            split = make_or_load_split(
                study_path, n_samples=len(X), n_test=20, n_val=20
            )

        for qoi_name, qoi_info in QOI_COLS.items():
            out = os.path.join(study_path, qoi_name)
            json_path = os.path.join(out, f"hyperparams_{qoi_name}.json")

            if args.plot_only:
                if not os.path.exists(json_path):
                    print(f"  skip {qoi_name}: no {json_path} (run first)")
                    continue
                with open(json_path) as f:
                    record = json.load(f)
                best_type = record["best"]
                if not os.path.exists(
                    _convergence_data_path(out, best_type, qoi_name)
                ):
                    print(f"  skip {qoi_name}: no saved convergence data")
                    continue
                plot_convergence(out, best_type, qoi_name, qoi_info["ylabel"])
                print(f"  {qoi_name}: replotted ({best_type})")
            else:
                y = Y_full[qoi_name].values.reshape(-1, 1) * -1
                best_type, best_params = select_or_load_surrogate(
                    X, y, split, qoi_name, out
                )
                with open(json_path) as f:
                    record = json.load(f)
                run_optimization(
                    X,
                    y,
                    model_type=best_type,
                    params=best_params,
                    n_runs=50,
                    max_iters=1000,
                    bootstrap_size=250,
                    out_folder=out,
                    qoi_name=qoi_name,
                    qoi_ylabel=qoi_info["ylabel"],
                )

            selection_rows.append(
                {
                    "level": label,
                    "qoi_name": qoi_name,
                    "best": best_type,
                    "val_mse": record[best_type]["val_mse"],
                    "val_nrmse_amp_pct": record[best_type][
                        "val_nrmse_amp_pct"
                    ],
                    "val_nrmse_mean_pct": record[best_type][
                        "val_nrmse_mean_pct"
                    ],
                    "best_params": record["best_params"],
                }
            )

    with open("surrogate_selection.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "level",
                "qoi_name",
                "best",
                "val_mse",
                "val_nrmse_amp_pct",
                "val_nrmse_mean_pct",
                "best_params",
            ],
        )
        writer.writeheader()
        writer.writerows(selection_rows)
    print("\nWrote surrogate_selection.csv")

    plot_design_composition(studies, "design_composition_barplot")
    print("Wrote design_composition_barplot.png")
