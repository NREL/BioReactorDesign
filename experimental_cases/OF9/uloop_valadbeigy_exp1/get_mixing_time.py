"""Volume-averaged dye tracer (Z.liquid) in the bottom U-bend box vs time.

For every time folder this reads the ``Z.liquid`` with  BiRD
then computes the cell-volume-weighted average over the
cells whose centres lie inside the box

    x in [-0.01, 0.01],  y in [-0.1, 0.1],  z in [-0.1, 0.1]   [m]

and plots that average versus time.
"""

import os

import numpy as np
from prettyPlot.plotting import plt, pretty_labels

from bird.utilities.ofio import (
    get_case_times,
    read_cell_centers,
    read_cell_volumes,
    read_field,
)

CASE = os.path.dirname(os.path.abspath(__file__))

# Averaging box [m]. The mesh is y-up after the presteps transformPoints.
X_BOUNDS = (-0.01, 0.01)
Y_BOUNDS = (-0.1, 0.1)
Z_BOUNDS = (-0.1, 0.1)

# Mixing-time criterion.
BAND = 0.05  # within +/-5% of the final well-mixed value
TAIL_WINDOW = 0.5  # s, tail used to define the final value


def read_dye_start():
    """Dye injection start time [s], read from constant/globalVars."""
    with open(os.path.join(CASE, "constant", "globalVars")) as f:
        for line in f:
            if line.startswith("dyeStart"):
                return float(line.split()[1].rstrip(";"))
    return 1.0


def box_mask(cell_centers):
    """Boolean mask of cells whose centres lie inside the averaging box."""
    x, y, z = cell_centers[:, 0], cell_centers[:, 1], cell_centers[:, 2]
    return (
        (x >= X_BOUNDS[0])
        & (x <= X_BOUNDS[1])
        & (y >= Y_BOUNDS[0])
        & (y <= Y_BOUNDS[1])
        & (z >= Z_BOUNDS[0])
        & (z <= Z_BOUNDS[1])
    )


def box_volume_average(z_field, cell_volumes, mask):
    """Cell-volume-weighted average of z_field over the masked cells.

    read_field returns a bare float for a uniform OpenFOAM field (e.g. the dye
    tracer before injection starts); the volume average is then exactly that
    value. Otherwise it is the volume-weighted mean over the box cells.
    """
    if np.ndim(z_field) == 0:
        return float(z_field)
    else:
        vol = cell_volumes[mask]
        return float(np.sum(z_field[mask] * vol) / np.sum(vol))


def mixing_time(t_arr, z_arr, t_start, continuous=True):
    """Mixing time from the box-averaged dye signal.

    The final well-mixed value is the mean of the signal over the last
    TAIL_WINDOW seconds. The mixing time is the interval from dye injection
    (t_start) to the last instant the signal leaves the +/-BAND envelope of
    that final value (after which it stays inside for good).

    Parameters
    ----------
    t_arr : array-like
        Time array.
    z_arr : array-like
        Box-averaged dye signal.
    t_start : float
        Injection time.
    continuous : bool, optional
        If True, linearly interpolates between the last timestep outside the
        band and the first timestep inside to find the exact crossing time.

    Returns
    -------
    t_mix : float
        Mixing time measured from injection [s].
    t_settle : float
        Absolute simulation time at which the signal settles [s].
    z_final : float
        Final well-mixed box-averaged value.
    """
    # Calculate final value and allowable band
    z_final = float(np.mean(z_arr[t_arr >= t_arr[-1] - TAIL_WINDOW]))
    band = BAND * abs(z_final)

    post = t_arr >= t_start
    outside = post & (np.abs(z_arr - z_final) > band)

    if not np.any(outside):
        # already within the band from injection onward
        t_settle = t_start
    else:
        last_out = np.nonzero(outside)[0][-1]

        # Check if we have a subsequent point to interpolate with
        if last_out + 1 < len(t_arr):
            if continuous:
                # Extract time and Z values for the crossing interval
                t0, t1 = t_arr[last_out], t_arr[last_out + 1]
                z0, z1 = z_arr[last_out], z_arr[last_out + 1]

                # Determine which boundary of the band was crossed
                if z0 > z_final:
                    z_target = z_final + band  # Crossed the top boundary
                else:
                    z_target = z_final - band  # Crossed the bottom boundary

                # Linearly interpolate to find the exact time t_settle at z_target
                if z1 != z0:  # Safety check to prevent division by zero
                    t_settle = t0 + (t1 - t0) * (z_target - z0) / (z1 - z0)
                else:
                    t_settle = t1
            else:
                # Original discrete behavior
                t_settle = t_arr[last_out + 1]
        else:
            # The signal was outside the band up to the very last recorded timestep
            t_settle = t_arr[last_out]

    return t_settle - t_start, t_settle, z_final


if __name__ == "__main__":
    os.makedirs(os.path.join(CASE, "Figures"), exist_ok=True)

    # Geometry is time-independent: read the cell centres and cell volumes once
    # and keep them in the shared field_dict cache.
    cell_centers, geom = read_cell_centers(CASE)
    n_cells = cell_centers.shape[0]
    cell_volumes, geom = read_cell_volumes(CASE, field_dict=geom)

    mask = box_mask(cell_centers)
    if mask.sum() == 0:
        raise RuntimeError("averaging box contains no cell centres")

    times_float, times_str = get_case_times(CASE)

    t_list, z_list = [], []
    for t_val, t_str in zip(times_float, times_str):
        try:
            # Fresh field_dict per time so Z.liquid is never served stale.
            z_field, _ = read_field(CASE, t_str, "Z.liquid", n_cells=n_cells)
        except FileNotFoundError:
            continue
        t_list.append(t_val)
        z_list.append(box_volume_average(z_field, cell_volumes, mask))

    order = np.argsort(t_list)
    t_arr = np.asarray(t_list)[order]
    z_arr = np.asarray(z_list)[order]

    np.savetxt(
        os.path.join(CASE, "Z_box_average.dat"),
        np.column_stack([t_arr, z_arr]),
        header="time[s]  volAvg(Z.liquid)_box",
    )

    dye_start = read_dye_start()
    t_mix, t_settle, z_final = mixing_time(t_arr, z_arr, dye_start)
    t_mix_disc, t_settle_disc, z_final_disc = mixing_time(
        t_arr, z_arr, dye_start, continuous=False
    )

    with open(os.path.join(CASE, "mix_time.txt"), "w") as f:
        f.write(f"Continuous: {t_mix:.4f}\n")
        f.write(f"Discrete: {t_mix_disc:.4f}\n")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_arr, z_arr, color="k")
    ax.axhline(z_final, color="b", ls="--", label=r"$Z_{final}$")
    ax.axhspan(
        (1 - BAND) * z_final, (1 + BAND) * z_final, color="b", alpha=0.15
    )
    ax.axvline(
        t_settle, color="r", ls=":", label=f"$t_{{mix}}$={t_mix_disc:.2f} s"
    )
    ax.set_xlim(left=dye_start)
    pretty_labels("time [s]", r"box-averaged $Z_{liquid}$ [-]", 14, ax=ax)
    ax.legend()
    fig.savefig(
        os.path.join(CASE, "Figures", "Z_box_average.png"),
        dpi=150,
        bbox_inches="tight",
    )

    print(f"box cells             : {int(mask.sum())}")
    print(f"time folders averaged : {len(t_arr)}")
    print(f"final well-mixed Z discrete     : {z_final_disc:.6g}")
    print(f"final well-mixed Z continous    : {z_final:.6g}")
    print(f"dye injection start    : {dye_start:.3f} s")
    print(
        f"mixing time discrete (+/-5%)    : {t_mix_disc:.3f} s  (settles at t={t_settle_disc:.3f} s)"
    )
    print(
        f"mixing time continuous (+/-5%)    : {t_mix:.3f} s  (settles at t={t_settle:.3f} s)"
    )
    print(
        "wrote mix_time.txt, Z_box_average.dat and Figures/Z_box_average.png"
    )
