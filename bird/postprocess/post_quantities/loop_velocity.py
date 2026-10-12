import numpy as np

from bird import logger
from bird.meshing.block_rect_mesh import from_block_rect_to_seg
from bird.utilities.ofio import read_cell_volumes, read_field

from ._cell_filter import _weighted_average


def _normalize(vector: np.ndarray) -> np.ndarray:
    """Unit vector along ``vector`` (raises on a zero vector)."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Cannot normalize a zero-length direction vector")
    return vector / norm


def _report_coverage(covered: np.ndarray) -> None:
    """Warn about the fraction of uncovered cells; list their ids in debug."""
    n_cells = len(covered)
    n_uncovered = int(np.count_nonzero(~covered))
    if n_uncovered:
        logger.warning(
            f"{n_uncovered}/{n_cells} cells "
            f"({100.0 * n_uncovered / n_cells:.2f}%) have no loop direction "
            f"and are excluded from the loop velocity"
        )
        logger.debug(
            f"Uncovered cell ids: {np.flatnonzero(~covered).tolist()}"
        )


def build_loop_direction_field(
    cell_centers: np.ndarray, boxes: list[dict]
) -> np.ndarray:
    """Per-cell loop-direction field from labelled axis-aligned boxes.

    Each cell inherits the (unit) direction of the box that contains its centre.
    Cells in no box keep ``NaN`` (excluded from the average); a cell in more than
    one box is ambiguous and raises.

    Parameters
    ----------
    cell_centers: np.ndarray
        Cell centres, shape ``(N, 3)``
    boxes: list[dict]
        List of ``{"min": [x, y, z], "max": [x, y, z], "direction": [dx, dy, dz]}``;
        ``direction`` is normalized internally

    Returns
    ----------
    direction_field: np.ndarray
        Direction field, shape ``(N, 3)``, ``NaN`` where uncovered
    """
    cell_centers = np.asarray(cell_centers, dtype=float)
    n_cells = len(cell_centers)
    direction_field = np.full((n_cells, 3), np.nan)
    coverage_count = np.zeros(n_cells, dtype=int)

    for box in boxes:
        box_min = np.asarray(box["min"], dtype=float)
        box_max = np.asarray(box["max"], dtype=float)
        direction = _normalize(np.asarray(box["direction"], dtype=float))
        inside = np.all(
            (cell_centers >= box_min) & (cell_centers <= box_max), axis=1
        )
        coverage_count += inside
        direction_field[inside] = direction

    n_overlap = int(np.count_nonzero(coverage_count > 1))
    if n_overlap:
        raise ValueError(
            f"{n_overlap} cells fall in more than one box; loop-direction "
            f"boxes must not overlap"
        )

    covered = coverage_count > 0
    if not np.any(covered):
        raise ValueError(
            "No cell falls within any loop box; the boxes and the mesh are in "
            "different coordinate frames. For a block-rectangular case this "
            "usually means the 'rescale' parameter is missing from mesh.json "
            "(assumed 1.0) - supply the rescale used to build the mesh."
        )

    _report_coverage(covered)
    return direction_field


def build_loop_direction_field_from_path(
    cell_centers: np.ndarray, path_points: np.ndarray, max_dist: float
) -> np.ndarray:
    """Per-cell loop-direction field from a centerline polyline.

    Each cell within ``max_dist`` of the polyline gets the unit tangent of its
    nearest segment; cells farther away keep ``NaN`` (excluded). The polyline
    ordering sets the circulation sense.

    Parameters
    ----------
    cell_centers: np.ndarray
        Cell centres, shape ``(N, 3)``
    path_points: np.ndarray
        Ordered centerline vertices, shape ``(M, 3)``
    max_dist: float
        Cells beyond this distance from the path are left uncovered

    Returns
    ----------
    direction_field: np.ndarray
        Direction field, shape ``(N, 3)``, ``NaN`` where uncovered
    """
    cell_centers = np.asarray(cell_centers, dtype=float)
    path_points = np.asarray(path_points, dtype=float)
    if len(path_points) < 2:
        raise ValueError("path_points must contain at least two vertices")

    n_cells = len(cell_centers)
    best_distance = np.full(n_cells, np.inf)
    direction_field = np.full((n_cells, 3), np.nan)

    for start, end in zip(path_points[:-1], path_points[1:]):
        segment = end - start
        length_sq = float(segment @ segment)
        if length_sq == 0.0:
            continue
        tangent = segment / np.sqrt(length_sq)
        # Projection parameter of each cell onto the segment, clamped to [0, 1]
        param = np.clip(
            ((cell_centers - start) @ segment) / length_sq, 0.0, 1.0
        )
        projection = start + param[:, None] * segment
        distance = np.linalg.norm(cell_centers - projection, axis=1)
        closer = distance < best_distance
        best_distance[closer] = distance[closer]
        direction_field[closer] = tangent

    uncovered = best_distance > max_dist
    direction_field[uncovered] = np.nan
    if np.all(uncovered):
        raise ValueError(
            "No cell lies within max_dist of the path; the path and the mesh "
            "are in different coordinate frames, or max_dist is too small"
        )
    _report_coverage(~uncovered)
    return direction_field


def propose_loop_boxes_block_rect(
    mesh_geometry: dict, rescale: float | None = None
) -> list[dict]:
    """Candidate loop-direction boxes for a block-rectangular loop reactor.

    Builds one box per mesh segment (leg) from
    :func:`bird.meshing.block_rect_mesh.from_block_rect_to_seg`. Each box spans the
    leg along its axis, trimmed half a block at each end so adjacent legs do not
    overlap and the ambiguous junction cells stay uncovered. The proposed direction
    is each leg's own axis orientation (``end - start``).

    The mesh scale is never inferred: the box coordinates are scaled by ``rescale``,
    which the caller must set to the factor used to build the mesh (e.g. the
    ``transformPoints`` scale). A mismatch surfaces downstream as a zero-coverage
    error in :func:`build_loop_direction_field`.

    Parameters
    ----------
    mesh_geometry: dict
        The ``"Geometry"`` dict from the case ``mesh.json``
    rescale: float | None
        Uniform scale factor applied to the box coordinates; ``None`` assumes
        1.0 (base units)

    Returns
    ----------
    boxes: list[dict]
        Box list consumable by :func:`build_loop_direction_field`
    """
    factor = 1.0 if rescale is None else float(rescale)
    segment_data = from_block_rect_to_seg(mesh_geometry, rescale=False)
    segments = segment_data["segments"]
    block_size = segment_data["blocksize"]

    boxes = []
    for segment in segments.values():
        start = np.asarray(segment["start"], dtype=float)
        end = np.asarray(segment["end"], dtype=float)
        normal_dir = int(segment["normal_dir"])
        max_rad = float(segment["max_rad"])

        box_min = np.empty(3)
        box_max = np.empty(3)
        for axis in range(3):
            if axis == normal_dir:
                low = min(start[axis], end[axis]) + 0.5 * block_size[axis]
                high = max(start[axis], end[axis]) - 0.5 * block_size[axis]
            else:
                low = start[axis] - max_rad
                high = start[axis] + max_rad
            box_min[axis] = low
            box_max[axis] = high

        boxes.append(
            {
                "min": (box_min * factor).tolist(),
                "max": (box_max * factor).tolist(),
                "direction": _normalize(end - start).tolist(),
            }
        )
    return boxes


def _read_liquid_fraction(
    case_folder: str, time_folder: str, n_cells: int, field_dict: dict
) -> tuple[np.ndarray | float, dict]:
    """Liquid fraction from alpha.gas (1 - alpha.gas) or alpha.liquid."""
    try:
        alpha_gas, field_dict = read_field(
            case_folder, time_folder, "alpha.gas", n_cells, field_dict
        )
        return 1.0 - np.asarray(alpha_gas), field_dict
    except FileNotFoundError:
        pass
    try:
        alpha_liquid, field_dict = read_field(
            case_folder, time_folder, "alpha.liquid", n_cells, field_dict
        )
        return alpha_liquid, field_dict
    except FileNotFoundError:
        raise FileNotFoundError(
            "Need alpha.gas or alpha.liquid to average over the liquid"
        )


def compute_loop_velocity(
    case_folder: str,
    time_folder: str,
    loop_direction_field: np.ndarray,
    volume_time: str | None = None,
    field_dict: dict | None = None,
) -> tuple[float, dict]:
    r"""Loop velocity: liquid velocity projected on the loop direction.

    .. math::
       \frac{\int_{V_{\rm loop}} \alpha_{\rm liq}\, (\mathbf{U}_{\rm liq} \cdot
       \hat{\mathbf{e}}_{\rm loop})\, dV}{\int_{V_{\rm loop}} \alpha_{\rm liq}\, dV}

    Averaged over the liquid on the covered cells. Positive follows the prescribed
    circulation, negative is reversed.

    Parameters
    ----------
    case_folder: str
        Path to case folder
    time_folder: str
        Name of the time folder to analyze
    loop_direction_field: np.ndarray
        ``(N, 3)`` field from a builder, ``NaN`` on uncovered cells
    volume_time : str | None
        Time folder to read to get the cell volumes.
        If None, finds volume time automatically
    field_dict : dict
        Dictionary of fields used to avoid rereading the same fields to calculate different quantities

    Returns
    ----------
    loop_velocity: float
        Volume averaged loop velocity, in :math:`m.s^{-1}`
    field_dict : dict
        Dictionary of fields read
    """
    if field_dict is None:
        field_dict = {}

    loop_direction_field = np.asarray(loop_direction_field, dtype=float)
    n_cells = len(loop_direction_field)
    covered = ~np.isnan(loop_direction_field).any(axis=1)
    if not np.any(covered):
        raise ValueError("The loop-direction field covers no cell")

    u_liquid, field_dict = read_field(
        case_folder, time_folder, "U.liquid", n_cells, field_dict
    )
    # A uniform vector field is read as shape (3,); broadcast to one per cell
    if np.ndim(u_liquid) == 1:
        u_liquid = np.broadcast_to(u_liquid, (n_cells, 3))

    cell_volume, field_dict = read_cell_volumes(
        case_folder, volume_time, n_cells, field_dict
    )
    alpha_liquid, field_dict = _read_liquid_fraction(
        case_folder, time_folder, n_cells, field_dict
    )

    # Liquid velocity projected on the loop direction, averaged over the liquid
    weights = np.asarray(cell_volume) * np.asarray(alpha_liquid)
    if weights.ndim == 0:
        weights = np.full(n_cells, float(weights))
    projected = np.sum(
        u_liquid[covered] * loop_direction_field[covered], axis=1
    )
    loop_velocity = _weighted_average(projected, weights[covered])

    return float(loop_velocity), field_dict
