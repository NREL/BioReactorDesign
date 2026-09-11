import argparse

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from prettyPlot.plotting import plt

# fraction-along-branch of each spot (matches randsim.optimization_setup)
BRANCH_SPOTS = {
    0: np.linspace(0.2, 0.8, 4),
    1: np.linspace(0.2, 0.8, 3),
    2: np.linspace(0.2, 0.8, 4),
}

# bottom-face leg endpoints in (x, z) block coords (matches mesh.json Fluids)
BRANCH_ENDS = {
    0: ((0.0, 0.0), (9.0, 0.0)),
    1: ((9.0, 0.0), (9.0, 4.0)),
    2: ((9.0, 4.0), (0.0, 4.0)),
}

CHOICE_COLOR = {1: "tab:blue", 0: "red", 2: "black"}
CHOICE_NAME = {1: "Sparger", 0: "Mixer", 2: "Wall"}


def spot_positions() -> list[tuple[float, float]]:
    """(x, z) of the 11 spots, ordered as the design vector [b0, b1, b2]."""
    positions = []
    for branch_id in (0, 1, 2):
        (start_x, start_z), (end_x, end_z) = BRANCH_ENDS[branch_id]
        for frac in BRANCH_SPOTS[branch_id]:
            positions.append(
                (
                    start_x + frac * (end_x - start_x),
                    start_z + frac * (end_z - start_z),
                )
            )
    return positions


def plot_schematic(
    design_x: np.ndarray, out_path: str, title: str | None = None
) -> None:
    """Draw the bottom-face schematic for one design vector (length 11)."""
    design_x = np.asarray(design_x, dtype=int)
    if len(design_x) != 11:
        raise ValueError(f"expected 11 design values, got {len(design_x)}")
    positions = spot_positions()

    fig, ax = plt.subplots(figsize=(7, 4))
    # legs as thick rounded grey lines (the tube centerlines)
    for branch_id in (0, 1, 2):
        (start_x, start_z), (end_x, end_z) = BRANCH_ENDS[branch_id]
        ax.plot(
            [start_x, end_x],
            [start_z, end_z],
            color="lightgray",
            linewidth=16,
            solid_capstyle="round",
            zorder=1,
        )
    # placement spots
    for (spot_x, spot_z), choice in zip(positions, design_x):
        ax.add_patch(
            Circle(
                (spot_x, spot_z),
                0.32,
                facecolor=CHOICE_COLOR[int(choice)],
                edgecolor="k",
                linewidth=1.2,
                zorder=3,
            )
        )

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.0, 10.0)
    ax.set_ylim(-1.5, 5.5)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=CHOICE_COLOR[choice],
            markeredgecolor="k",
            markersize=12,
            label=CHOICE_NAME[choice],
        )
        for choice in (1, 0, 2)
    ]
    # place the legend in the open interior of the U so it clears the legs
    ax.legend(handles=legend_handles, loc="center", frameon=False)
    if title is not None:
        ax.set_title(title, fontsize=13)

    fig.savefig(f"{out_path}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
    plt.close(fig)


def design_from_csv(csv_path: str) -> np.ndarray:
    """Read x0..x10 from a best_bootstrap_solution CSV."""
    row = pd.read_csv(csv_path).iloc[0]
    return np.array([int(row[f"x{i}"]) for i in range(11)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a design schematic")
    parser.add_argument(
        "-i", "--input", required=True, help="best_bootstrap_solution CSV"
    )
    parser.add_argument(
        "-o", "--out", default="optimum_schematic", help="output path (no ext)"
    )
    parser.add_argument("-t", "--title", default=None)
    args = parser.parse_args()

    plot_schematic(design_from_csv(args.input), args.out, title=args.title)
    print(f"Wrote {args.out}.png / .pdf")
