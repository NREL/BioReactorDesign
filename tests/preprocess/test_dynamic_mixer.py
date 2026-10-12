import os
import tempfile
from pathlib import Path

import numpy as np

from bird.meshing.block_rect_mesh import from_block_rect_to_seg
from bird.preprocess.dynamic_mixer.mixer import ActuatorMixer
from bird.preprocess.dynamic_mixer.mixing_fvModels import *
from bird.utilities.parser import parse_json


def test_expl_list():
    BIRD_PRE_DYNMIX_TEMP_DIR = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "bird",
        "preprocess",
        "dynamic_mixer",
        "mixing_template",
    )
    input_dict = parse_json(
        os.path.join(BIRD_PRE_DYNMIX_TEMP_DIR, "expl_list", "mixers.json")
    )

    # Output to temporary directory and delete when done
    with tempfile.TemporaryDirectory() as tmpdirname:
        write_fvModel(input_dict, output_folder=tmpdirname)

    # Output to temporary directory and delete when done
    with tempfile.TemporaryDirectory() as tmpdirname:
        write_fvModel(input_dict, output_folder=tmpdirname, force_sign=True)


def test_loop_list():
    BIRD_PRE_DYNMIX_TEMP_DIR = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "bird",
        "preprocess",
        "dynamic_mixer",
        "mixing_template",
    )
    input_dict = parse_json(
        os.path.join(
            BIRD_PRE_DYNMIX_TEMP_DIR, "loop_reactor_list", "mixers.json"
        )
    )

    # Output to temporary directory and delete when done
    with tempfile.TemporaryDirectory() as tmpdirname:
        write_fvModel(input_dict, output_folder=tmpdirname)

    # Output to temporary directory and delete when done
    with tempfile.TemporaryDirectory() as tmpdirname:
        write_fvModel(input_dict, output_folder=tmpdirname, force_sign=True)


def test_ActuatorMixer():
    geom = {
        "OverallDomain": {
            a: {"size_per_block": 1.0, "rescale": 2.76}
            for a in ("x", "y", "z")
        },
        "Fluids": [[[0, 0, 0], [9, 0, 0]]],
    }
    g = from_block_rect_to_seg(geom)

    # loop mode: radius is a fraction of the branch cross-section
    m = ActuatorMixer()
    m.update_from_loop_dict(
        {
            "branch_id": 0,
            "frac_space": 0.4,
            "radius": 0.4,
            "sign": "+",
            "swirl_sign": "+",
            "Vtip": 1.5,
            "Np": 6,
            "sigma": 0.35,
            "start_time": 3,
        },
        g,
    )
    assert m.ready
    assert m.normal_dir == 0
    assert abs(m.R - 0.4 * 2.76) < 1e-9  # frac * mean transverse block size
    assert (m.Vtip, m.Np, m.sigma) == (1.5, 6, 0.35)
    assert m.sign == "+" and m.swirl_sign == "+"
    # position = segment start + frac*conn, block size = 2.76
    assert abs(m.x - (0.5 * 2.76 + 0.4 * 9 * 2.76)) < 1e-6
    assert abs(m.y - 0.5 * 2.76) < 1e-6

    # explicit mode: radius is absolute
    m2 = ActuatorMixer()
    m2.update_from_expl_dict(
        {
            "x": 0.1,
            "y": 0.2,
            "z": 0.3,
            "normal_dir": 1,
            "radius": 0.05,
            "sign": "-",
        }
    )
    assert m2.ready and abs(m2.R - 0.05) < 1e-12 and m2.normal_dir == 1

    # missing sign leaves the mixer not ready
    m3 = ActuatorMixer()
    m3.update_from_expl_dict(
        {"x": 0.1, "y": 0.2, "z": 0.3, "normal_dir": 1, "radius": 0.05}
    )
    assert not m3.ready


def test_write_fvModel_ball():
    base = {
        "Meshing": {"Blockwise": {"x": 10, "y": 10, "z": 10}},
        "Geometry": {
            "OverallDomain": {
                a: {"nblocks": 10, "size_per_block": 1.0, "rescale": 2.76}
                for a in ("x", "y", "z")
            },
            "Fluids": [[[0, 0, 0], [9, 0, 0]]],
        },
        "volumetric_source": "ball",
        "mixers": [
            {
                "branch_id": 0,
                "frac_space": 0.5,
                "radius": 0.4,
                "sign": "+",
                "swirl_sign": "+",
                "Vtip": 1.5,
                "Np": 6,
                "sigma": 0.35,
                "power": 3000,
                "start_time": 1,
            }
        ],
    }

    # swirl endpoint: from_Np_Vtip + axial_and_swirl
    d = dict(base, power="from_Np_Vtip", momentum_source="axial_and_swirl")
    with tempfile.TemporaryDirectory() as tmpdirname:
        write_fvModel(d, output_folder=tmpdirname)
        txt = Path(tmpdirname, "fvModels").read_text()
    assert "// ===== ball mixer =====" in txt
    assert "dynamicMix_util" not in txt  # Newton solve is inlined
    assert "16.0*6" in txt and "pow(pi,4.0)" in txt  # Np/Vtip drive
    assert "Qsw" in txt  # swirl torque present
    # exact conservation: runtime-summed normalisers
    assert "reduce(Sax, sumOp<scalar>());" in txt
    assert "reduce(Sth, sumOp<scalar>());" in txt
    assert "Tax/Sax" in txt and "Qsw/Sth" in txt

    # axial + from_P endpoint
    d = dict(base, power="from_P", momentum_source="axial")
    with tempfile.TemporaryDirectory() as tmpdirname:
        write_fvModel(d, output_folder=tmpdirname)
        txt = Path(tmpdirname, "fvModels").read_text()
    assert "4.0*3000/(rhoM*area)" in txt  # P drive
    assert "Qsw" not in txt and "Sth" not in txt  # no swirl
