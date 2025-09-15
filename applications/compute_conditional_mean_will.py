import os

import numpy as np

from bird import logger
from bird.postprocess.post_quantities import _get_ind_slice
from bird.utilities.mathtools import conditional_average
from bird.utilities.ofio import get_case_times, read_cell_centers, read_field

logger.setLevel("DEBUG")

from pathlib import Path


def height2str(height: float) -> str:
    """
    Convert height value to a str
    This is used as a key to access specific height info from dicts
    """
    return f"{height:.2g}"


def compute_radius_field(
    cell_centers: np.ndarray, vert_ind: int
) -> np.ndarray:
    """
    Compute radius from cell centres. The radius goes negative to make it look like brooks and chen
    You might want to review this
    """
    if vert_ind == 0:
        radius = np.sqrt(cell_centers[:, 1] ** 2 + cell_centers[:, 2] ** 2)
        ind_min = np.argwhere(cell_centers[:, 1] < 0)
        radius[ind_min] = -1 * radius[ind_min]
    elif vert_ind == 1:
        radius = np.sqrt(cell_centers[:, 0] ** 2 + cell_centers[:, 2] ** 2)
        ind_min = np.argwhere(cell_centers[:, 0] < 0)
        radius[ind_min] = -1 * radius[ind_min]
    elif vert_ind == 2:
        radius = np.sqrt(cell_centers[:, 0] ** 2 + cell_centers[:, 1] ** 2)
        ind_min = np.argwhere(cell_centers[:, 0] < 0)
        radius[ind_min] = -1 * radius[ind_min]
    else:
        raise ValueError(f"vertical index must be 0, 1 or 2, got {vert_ind}")

    return radius


def get_heights_ind(
    case_folder: str,
    heights: list[float],
    vert_ind: int,
) -> list[np.ndarray]:
    """
    get the cell center indexes that correspond to the height you care about
    Uses the bird _get_ind_slice function looped over all the heights you want
    """
    ind_heights = {}
    for height in heights:
        ind_heights[height2str(height)], _ = _get_ind_slice(
            case_folder, location=height, direction=vert_ind
        )
    return ind_heights


def radial_mean(
    case_folder=None,
    heights: list[float] | None = None,
    field_names: list[str] = ["CO2.liquid"],
    vert_ind: int | None = 1,
    window_ave=1,
    n_bins=32,
) -> dict:
    """
    Compute radial conditional average
    The conditional average is averaged over space and over time (using the window_ave number of times)
    """
    if case_folder is None:
        case_folder = os.path.join(
            Path(__file__).parent,
            "..",
            "bird",
            "postprocess",
            "data_conditional_mean",
        )

    logger.info(f"Case : {case_folder}")

    # Get all time folders
    time_float_sorted, time_str_sorted = get_case_times(case_folder)

    # Get cell centers
    cell_centers, _ = read_cell_centers(case_folder)
    window_ave = min(window_ave, len(time_str_sorted))

    # Get the cell indices that correspond to the height you care about
    if heights is not None:
        if vert_ind is None:
            msg = "Assuming vertical direction is y"
            msg += f"\nIf that is not the case, change the parameter vert_ind (currently {vert_ind})"
            logger.warning(msg)
            vert_ind = 1
        ind_heights = get_heights_ind(
            case_folder=case_folder,
            heights=heights,
            vert_ind=vert_ind,
        )

    # Create a radius field (to condition against)
    radius = compute_radius_field(cell_centers, vert_ind)

    # Setup the structure in which results are saved
    fields_conds = {}
    fields_conds_tmp = {}
    for field_name in field_names:
        fields_conds[field_name] = {}
        if heights is not None:
            radius_axis = {}
            for height in heights:
                fields_conds[field_name][height2str(height)] = {}

    # loop over the time for which window averaging is performed
    for i_ave in range(window_ave):
        time_folder = time_str_sorted[-i_ave - 1]
        logger.info(f"\tReading Time : {time_folder}")

        # Get the filename of files that need to be read
        field_file = []
        for field_name in field_names:
            field_file.append(
                os.path.join(case_folder, time_folder, field_name)
            )

        # Read all the fields you want
        for filename, field_name in zip(field_file, field_names):
            val_dict = {}
            field_tmp, _ = read_field(case_folder, time_folder, field_name)
            if len(field_tmp.shape) == 2 and field_tmp.shape[1] == 3:
                # You read a velocity I'm assuming you need the axial one
                field_tmp = field_tmp[:, vert_ind]

            if heights is None:
                radius_axis, fields_conds_tmp = conditional_average(
                    radius, field_tmp, nbins=n_bins
                )
            else:
                # Filter fields so we only look at the heights you care about
                for height in heights:
                    inds = ind_heights[height2str(height)]
                    (
                        radius_axis[height2str(height)],
                        fields_conds_tmp[height2str(height)],
                    ) = conditional_average(
                        radius[inds], field_tmp[inds], nbins=n_bins
                    )

            # Do window averaging (accumulate window averaged quantity into fields_conds)
            if i_ave == 0:
                if heights is None:
                    fields_conds[field_name]["val"] = (
                        fields_conds_tmp / window_ave
                    )
                    fields_cond[sfield_name]["radius"] = radius_axis
                else:
                    for height in heights:
                        fields_conds[field_name][height2str(height)]["val"] = (
                            fields_conds_tmp[height2str(height)] / window_ave
                        )
                        fields_conds[field_name][height2str(height)][
                            "radius"
                        ] = radius_axis[height2str(height)]

            else:
                if heights is None:
                    fields_conds[field_name]["val"] += (
                        fields_conds_tmp / window_ave
                    )
                else:
                    for height in heights:
                        fields_conds[field_name][height2str(height)][
                            "val"
                        ] += (
                            fields_conds_tmp[height2str(height)] / window_ave
                        )

    return fields_conds


def make_plot(fields_cond: dict, heights: list[float], field_names=list[str]):

    from prettyPlot.plotting import plt, pretty_labels

    fig, axs = plt.subplots(
        nrows=len(field_names),
        ncols=len(heights),
        figsize=(4 * len(heights), 4 * len(field_names)),
    )

    for ifield, field_name in enumerate(field_names):
        for iheight, height in enumerate(heights):
            height_name = height2str(height)
            if len(field_names) == 1:
                ax = axs[iheight]
            else:
                ax = axs[ifield, iheight]

            ax.plot(
                fields_cond[field_name][height_name]["radius"],
                fields_cond[field_name][height_name]["val"],
                color="k",
                linewidth=3,
            )
            pretty_labels(
                "radius[m]",
                f"{field_name}",
                title=f"z={height_name}m",
                fontname="Times",
                fontsize=16,
                ax=ax,
                grid=False,
            )

    plt.show()


if __name__ == "__main__":

    # compute
    fields_cond = radial_mean(
        case_folder=None,
        heights=[5.2, 6.2, 6.3],
        field_names=["U.liquid", "alpha.gas"],
        vert_ind=1,
        window_ave=2,
        n_bins=32,
    )

    # plot
    # use logger level info to avoid annoying debug messages from matlplotlib
    logger.setLevel("INFO")
    make_plot(
        fields_cond=fields_cond,
        heights=[5.2, 6.2, 6.3],
        field_names=["U.liquid", "alpha.gas"],
    )
