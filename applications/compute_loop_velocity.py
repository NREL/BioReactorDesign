import argparse
import csv
import json
import os

from bird import logger
from bird.postprocess.post_quantities import (
    build_loop_direction_field,
    build_loop_direction_field_from_path,
    compute_loop_velocity,
    propose_loop_boxes_block_rect,
)
from bird.utilities.ofio import get_case_times, read_cell_centers


def build_direction_field(args, cell_centers):
    """Loop-direction field from the box, path, or block-rect proposer source."""
    if args.boxes is not None:
        with open(args.boxes) as f:
            boxes = json.load(f)
        return build_loop_direction_field(cell_centers, boxes)
    if args.path is not None:
        with open(args.path) as f:
            spec = json.load(f)
        return build_loop_direction_field_from_path(
            cell_centers, spec["points"], spec["max_dist"]
        )
    with open(args.mesh) as f:
        geometry = json.load(f)["Geometry"]
    boxes = propose_loop_boxes_block_rect(geometry, rescale=args.rescale)
    return build_loop_direction_field(cell_centers, boxes)


def main():
    parser = argparse.ArgumentParser(description="Loop (circulation) velocity")
    parser.add_argument(
        "-c", "--case", type=str, default=".", help="case folder"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "-b", "--boxes", type=str, help="JSON list of {min, max, direction}"
    )
    source.add_argument(
        "-p",
        "--path",
        type=str,
        help='JSON {"points": [[x,y,z], ...], "max_dist": float}',
    )
    source.add_argument(
        "-m", "--mesh", type=str, help="mesh.json for the block-rect proposer"
    )
    parser.add_argument(
        "-r",
        "--rescale",
        type=float,
        default=None,
        help="rescale factor for the proposer (default 1.0)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="output CSV (default <case>/Uliq_loop.csv)",
    )
    args = parser.parse_args()

    cell_centers, _ = read_cell_centers(args.case)
    direction_field = build_direction_field(args, cell_centers)

    time_values, time_names = get_case_times(args.case, remove_zero=True)
    output = args.output or os.path.join(args.case, "Uliq_loop.csv")
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "U_loop"])
        for time_value, time_name in zip(time_values, time_names):
            # fresh cache per time (the caller must reset it)
            u_loop, _ = compute_loop_velocity(
                args.case, time_name, direction_field, field_dict={}
            )
            writer.writerow([time_value, u_loop])
            logger.info(f"t={time_value:.4g}: U_loop={u_loop:.6g}")
    logger.info(f"Wrote {output}")


if __name__ == "__main__":
    main()
