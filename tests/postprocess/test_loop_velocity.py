import json
import os
import tempfile
from pathlib import Path

import numpy as np

from bird.postprocess.post_quantities import (
    build_loop_direction_field,
    build_loop_direction_field_from_path,
    compute_loop_velocity,
    propose_loop_boxes_block_rect,
)


def write_min_case(root, u_vectors, cell_volumes, alpha_gas=None):
    """Minimal case with a vector U.liquid, scalar V and optional alpha.gas."""
    os.makedirs(os.path.join(root, "0"), exist_ok=True)

    def write_field(name, foam_class, body):
        with open(os.path.join(root, "0", name), "w") as f:
            f.write("FoamFile\n{\n    format      ascii;\n")
            f.write(f"    class       {foam_class};\n")
            f.write(f"    object      {name};\n}}\n\n")
            f.write("dimensions      [0 0 0 0 0 0 0];\n\n")
            f.write(body)

    def scalar(values):
        entries = "\n".join(f"{v:.10g}" for v in values)
        return (
            "internalField   nonuniform List<scalar> \n"
            f"{len(values)}\n(\n{entries}\n)\n;\n"
        )

    def vector(vectors):
        entries = "\n".join(
            f"({x:.10g} {y:.10g} {z:.10g})" for x, y, z in vectors
        )
        return (
            "internalField   nonuniform List<vector> \n"
            f"{len(vectors)}\n(\n{entries}\n)\n;\n"
        )

    write_field("V", "volScalarField", scalar(cell_volumes))
    write_field("U.liquid", "volVectorField", vector(u_vectors))
    if alpha_gas is not None:
        write_field("alpha.gas", "volScalarField", scalar(alpha_gas))


def test_build_loop_direction_field():
    # Four cells: two in box A (+x), one in box B (+y), one in no box
    cell_centers = np.array(
        [[0.5, 0.0, 0.0], [1.5, 0.0, 0.0], [5.0, 5.0, 0.0], [9.0, 9.0, 9.0]]
    )
    boxes = [
        {"min": [0, -1, -1], "max": [2, 1, 1], "direction": [3, 0, 0]},
        {"min": [4, 4, -1], "max": [6, 6, 1], "direction": [0, 2, 0]},
    ]
    field = build_loop_direction_field(cell_centers, boxes)

    assert field.shape == (4, 3)
    # direction is normalized to a unit vector
    np.testing.assert_allclose(field[0], [1, 0, 0])
    np.testing.assert_allclose(field[1], [1, 0, 0])
    np.testing.assert_allclose(field[2], [0, 1, 0])
    # uncovered cell stays NaN
    assert np.all(np.isnan(field[3]))

    # a cell inside two boxes is ambiguous -> error
    overlap = [
        {"min": [0, -1, -1], "max": [2, 1, 1], "direction": [1, 0, 0]},
        {"min": [1, -1, -1], "max": [3, 1, 1], "direction": [0, 1, 0]},
    ]
    try:
        build_loop_direction_field(cell_centers, overlap)
        raised = False
    except ValueError:
        raised = True
    assert raised

    # boxes that match no cell (e.g. wrong coordinate frame / rescale) -> error
    far = [
        {
            "min": [100, 100, 100],
            "max": [101, 101, 101],
            "direction": [1, 0, 0],
        }
    ]
    try:
        build_loop_direction_field(cell_centers, far)
        raised_empty = False
    except ValueError:
        raised_empty = True
    assert raised_empty


def test_build_loop_direction_field_from_path():
    # Path: (0,0,0)->(1,0,0) [+x] then (1,0,0)->(1,1,0) [+y]
    path = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    cell_centers = np.array(
        [
            [0.5, 0.05, 0.0],  # near the +x segment
            [1.0, 0.5, 0.0],  # on the +y segment
            [0.5, 5.0, 0.0],  # far from the path
        ]
    )
    field = build_loop_direction_field_from_path(
        cell_centers, path, max_dist=0.5
    )

    np.testing.assert_allclose(field[0], [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(field[1], [0, 1, 0], atol=1e-12)
    assert np.all(np.isnan(field[2]))

    # every cell beyond max_dist (wrong frame) -> error
    try:
        build_loop_direction_field_from_path(
            np.array([[50.0, 50.0, 50.0]]), path, max_dist=0.5
        )
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_propose_loop_boxes_block_rect():
    template = os.path.join(
        Path(__file__).parent,
        " .. ".strip(),
        " .. ".strip(),
        "bird",
        "meshing",
        "block_rect_mesh_templates",
        "loopReactor",
        "input.json",
    )
    geometry = json.load(open(template))["Geometry"]
    # rescale=None -> base units (factor 1.0)
    boxes = propose_loop_boxes_block_rect(geometry, rescale=None)

    # two Fluids branches of 4 blocks -> 3 segments each -> 6 legs
    assert len(boxes) == 6

    # first leg: [0,0,0]->[9,0,0], centres at 0.5..9.5, trimmed half a block
    first = boxes[0]
    np.testing.assert_allclose(first["direction"], [1, 0, 0])
    np.testing.assert_allclose(first["min"], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(first["max"], [9.0, 1.0, 1.0])

    # a float rescale scales the box coordinates but not the direction
    scaled = propose_loop_boxes_block_rect(geometry, rescale=2.0)
    np.testing.assert_allclose(scaled[0]["min"], [2.0, 0.0, 0.0])
    np.testing.assert_allclose(scaled[0]["max"], [18.0, 2.0, 2.0])
    np.testing.assert_allclose(scaled[0]["direction"], [1, 0, 0])


def test_compute_loop_velocity():
    # Two cells pointing along +x, direction +x -> loop velocity = weighted |Ux|
    u_vectors = [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
    cell_volumes = [1.0, 3.0]
    direction_field = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    # all-liquid (alpha.gas = 0) -> volume-weighted mean = (2*1 + 4*3)/(1+3) = 3.5
    with tempfile.TemporaryDirectory() as tmp:
        write_min_case(tmp, u_vectors, cell_volumes, alpha_gas=[0.0, 0.0])
        v, _ = compute_loop_velocity(
            tmp, "0", direction_field, volume_time="0"
        )
    assert abs(v - 3.5) < 1e-10

    # a NaN direction excludes its cell: only the first cell counts -> 2.0
    with tempfile.TemporaryDirectory() as tmp:
        write_min_case(tmp, u_vectors, cell_volumes, alpha_gas=[0.0, 0.0])
        partial = np.array([[1.0, 0.0, 0.0], [np.nan, np.nan, np.nan]])
        v_partial, _ = compute_loop_velocity(
            tmp, "0", partial, volume_time="0"
        )
    assert abs(v_partial - 2.0) < 1e-10

    # reversed direction flips the sign
    with tempfile.TemporaryDirectory() as tmp:
        write_min_case(tmp, u_vectors, cell_volumes, alpha_gas=[0.0, 0.0])
        reversed_dir = np.array([[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        v_rev, _ = compute_loop_velocity(
            tmp, "0", reversed_dir, volume_time="0"
        )
    assert abs(v_rev + 3.5) < 1e-10

    # liquid weighting from alpha.gas: weights = vol*(1-alpha_gas) = [1, 1.5]
    with tempfile.TemporaryDirectory() as tmp:
        write_min_case(tmp, u_vectors, cell_volumes, alpha_gas=[0.0, 0.5])
        v_alpha, _ = compute_loop_velocity(
            tmp, "0", direction_field, volume_time="0"
        )
    assert abs(v_alpha - (2 * 1.0 + 4 * 1.5) / 2.5) < 1e-10

    # neither alpha.gas nor alpha.liquid present -> error
    with tempfile.TemporaryDirectory() as tmp:
        write_min_case(tmp, u_vectors, cell_volumes)
        try:
            compute_loop_velocity(tmp, "0", direction_field, volume_time="0")
            raised = False
        except FileNotFoundError:
            raised = True
    assert raised
