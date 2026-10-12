import os
import tempfile
from pathlib import Path

from bird.meshing.block_rect_mesh import from_block_rect_to_seg
from bird.preprocess.dynamic_mixer.io_fvModels import (
    write_end,
    write_preamble_ball,
    write_static_mixer_ball,
)
from bird.preprocess.dynamic_mixer.mixer import StaticMixer
from bird.preprocess.dynamic_mixer.mixing_fvModels import write_fvModel
from bird.utilities.parser import parse_json


def test_StaticMixer():
    geom = {
        "OverallDomain": {
            a: {"size_per_block": 1.0, "rescale": 2.76}
            for a in ("x", "y", "z")
        },
        "Fluids": [[[0, 0, 0], [9, 0, 0]]],
    }
    g = from_block_rect_to_seg(geom)

    # loop mode: radius is a fraction of the branch cross-section
    # (0.5 spans the whole tube)
    m = StaticMixer()
    m.update_from_loop_dict(
        {
            "branch_id": 0,
            "frac_space": 0.4,
            "radius": 0.5,
            "sign": "+",
            "swirl_sign": "+",
            "S": 0.35,
            "K": 0.5,
            "start_time": 3,
        },
        g,
    )
    assert m.ready
    assert m.normal_dir == 0
    assert abs(m.R - 0.5 * 2.76) < 1e-9  # frac * mean transverse block size
    assert (m.S, m.K) == (0.35, 0.5)
    assert m.sign == "+" and m.swirl_sign == "+"
    assert m.start_time == 3
    # position = segment start + frac*conn, block size = 2.76
    assert abs(m.x - (0.5 * 2.76 + 0.4 * 9 * 2.76)) < 1e-6
    assert abs(m.y - 0.5 * 2.76) < 1e-6

    # explicit mode: radius is absolute, defaults for S/K
    m2 = StaticMixer()
    m2.update_from_expl_dict(
        {
            "x": 0.1,
            "y": 0.2,
            "z": 0.3,
            "normal_dir": 1,
            "radius": 0.05,
            "sign": "-",
            "swirl_sign": "-",
        }
    )
    assert m2.ready and abs(m2.R - 0.05) < 1e-12 and m2.normal_dir == 1
    assert (m2.S, m2.K) == (0.35, 0.5)  # defaults
    assert m2.sign == "-" and m2.swirl_sign == "-"

    # missing sign leaves the mixer not ready
    m3 = StaticMixer()
    m3.update_from_expl_dict(
        {"x": 0.1, "y": 0.2, "z": 0.3, "normal_dir": 1, "radius": 0.05}
    )
    assert not m3.ready


def test_write_static_mixer_ball():
    m = StaticMixer()
    m.update_from_expl_dict(
        {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "normal_dir": 1,
            "radius": 0.05,
            "sign": "+",
            "swirl_sign": "+",
            "S": 0.35,
            "K": 0.5,
            "start_time": 1,
        }
    )
    assert m.ready

    with tempfile.TemporaryDirectory() as tmpdirname:
        write_preamble_ball(tmpdirname)
        write_static_mixer_ball(m, tmpdirname)
        write_end(tmpdirname)
        txt = Path(tmpdirname, "fvModels").read_text()

    assert "// ===== static mixer =====" in txt
    assert "dynamicMix_util" not in txt  # no external header
    assert "V2" not in txt  # passive: no Newton solve
    # passive loads (no tip speed / power)
    assert "const double Qsw = Snum*Rmix*rhoM*area*V1*V1;" in txt
    assert "const double Tls = 0.5*Kloss*rhoM*area*V1*V1;" in txt
    # activation gate inherited from the sensing block
    assert "if (V1 < 0.0) V1 = 0.0;" in txt
    # exact conservation: runtime-summed normalisers (velocity-weighted swirl)
    assert "reduce(Sax, sumOp<scalar>());" in txt
    assert "reduce(Ssw, sumOp<scalar>());" in txt
    assert "Ssw += alphaL[i]*g*rhoL[i]*ux*ux*rr*V[i];" in txt
    assert "const double A0 = Qsw/Ssw;" in txt
    # energy-neutral axial reaction f_cp ~ rho*ux*uth
    assert "const double fcp = A0*rhoL[i]*ux*uth*alphaL[i]*g;" in txt
    # normal_dir=1 -> theta_hat = (dz/rr, 0, -dx/rr); swirl on components 0 and 2
    # Usource -= B applies body force +B (adds B to the RHS), so the deposition
    # is emitted with -= and the pre-combined sign literals.
    assert "Usource[i][0] -= 1.0*fsw*V[i]*((dz)/rr);" in txt
    assert "Usource[i][2] -= 1.0*fsw*V[i]*((-dx)/rr);" in txt
    # axial reaction and viscous drag on the normal component (index 1)
    assert "Usource[i][1] -= -1.0*fcp*V[i];" in txt
    assert "Usource[i][1] -= -1.0*fvisc*V[i];" in txt


def test_write_static_mixer_ball_negative_orientation():
    # sign="-" and swirl_sign="-" must not emit "--1.0" (a decrement on a
    # literal, which is a C++ compile error)
    m = StaticMixer()
    m.update_from_expl_dict(
        {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "normal_dir": 1,
            "radius": 0.05,
            "sign": "-",
            "swirl_sign": "-",
            "S": 0.35,
            "K": 0.5,
            "start_time": 1,
        }
    )
    assert m.ready

    with tempfile.TemporaryDirectory() as tmpdirname:
        write_preamble_ball(tmpdirname)
        write_static_mixer_ball(m, tmpdirname)
        write_end(tmpdirname)
        txt = Path(tmpdirname, "fvModels").read_text()

    assert "--1.0" not in txt  # no decrement-on-literal
    # drag_ax and cp_th are +1.0 for the negative orientation
    assert "Usource[i][1] -= 1.0*fvisc*V[i];" in txt
    assert "Usource[i][1] -= 1.0*fcp*V[i];" in txt


def test_write_fvModel_static_mixers():
    # static-only, explicit placement: the ball path is auto-triggered by the
    # presence of the static_mixers list (no volumetric_source needed)
    d = {
        "static_mixers": [
            {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "normal_dir": 1,
                "radius": 0.05,
                "sign": "+",
                "swirl_sign": "+",
                "S": 0.35,
                "K": 0.5,
                "start_time": 1,
            }
        ]
    }
    with tempfile.TemporaryDirectory() as tmpdirname:
        write_fvModel(d, output_folder=tmpdirname)
        txt = Path(tmpdirname, "fvModels").read_text()
    assert txt.count("codedSource") == 1  # single preamble
    assert "// ===== static mixer =====" in txt
    assert "// ===== ball mixer =====" not in txt  # no dynamic block
    assert txt.rstrip().endswith("};")  # write_end closed the block

    # mixed dynamic + static, loop placement: both share one codedSource
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
                "start_time": 1,
            }
        ],
        "static_mixers": [
            {
                "branch_id": 0,
                "frac_space": 0.6,
                "radius": 0.5,
                "sign": "+",
                "swirl_sign": "+",
                "S": 0.35,
                "K": 0.5,
                "start_time": 1,
            }
        ],
    }
    with tempfile.TemporaryDirectory() as tmpdirname:
        write_fvModel(base, output_folder=tmpdirname)
        txt = Path(tmpdirname, "fvModels").read_text()
    assert txt.count("codedSource") == 1  # one shared block
    assert "// ===== ball mixer =====" in txt  # dynamic present
    assert "// ===== static mixer =====" in txt  # static present


def test_static_expl_list():
    template_dir = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "bird",
        "preprocess",
        "dynamic_mixer",
        "mixing_template",
    )
    d = parse_json(
        os.path.join(template_dir, "static_expl_list", "mixers.json")
    )
    with tempfile.TemporaryDirectory() as tmpdirname:
        write_fvModel(d, output_folder=tmpdirname)
        txt = Path(tmpdirname, "fvModels").read_text()
    assert "// ===== static mixer =====" in txt
    assert txt.rstrip().endswith("};")


def test_static_loop_list():
    template_dir = os.path.join(
        Path(__file__).parent,
        "..",
        "..",
        "bird",
        "preprocess",
        "dynamic_mixer",
        "mixing_template",
    )
    d = parse_json(
        os.path.join(template_dir, "static_loop_list", "mixers.json")
    )
    with tempfile.TemporaryDirectory() as tmpdirname:
        write_fvModel(d, output_folder=tmpdirname)
        txt = Path(tmpdirname, "fvModels").read_text()
    assert "// ===== static mixer =====" in txt
    assert txt.rstrip().endswith("};")
