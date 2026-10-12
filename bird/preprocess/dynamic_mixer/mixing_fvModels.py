from bird.meshing.block_rect_mesh import from_block_rect_to_seg
from bird.preprocess.dynamic_mixer.io_fvModels import *
from bird.preprocess.dynamic_mixer.mixer import (
    ActuatorMixer,
    Mixer,
    StaticMixer,
)


def check_input(input_dict):
    assert isinstance(input_dict, dict)
    mix_type = []
    for mix in input_dict.get("mixers", []):
        if "x" in mix:
            mix_type.append("expl")
        else:
            mix_type.append("loop")
    if "loop" in mix_type:
        assert "Geometry" in input_dict
        assert "OverallDomain" in input_dict["Geometry"]
        assert "x" in input_dict["Geometry"]["OverallDomain"]
        assert "y" in input_dict["Geometry"]["OverallDomain"]
        assert "z" in input_dict["Geometry"]["OverallDomain"]
        assert "size_per_block" in input_dict["Geometry"]["OverallDomain"]["x"]
        assert "Fluids" in input_dict["Geometry"]
        assert isinstance(input_dict["Geometry"]["Fluids"], list)
        assert isinstance(input_dict["Geometry"]["Fluids"][0], list)

    return mix_type


def check_static_input(input_dict):
    """Return the expl/loop type of each entry in the ``static_mixers`` list."""
    static_mix_type = []
    for mix in input_dict.get("static_mixers", []):
        static_mix_type.append("expl" if "x" in mix else "loop")
    if "loop" in static_mix_type:
        assert "Geometry" in input_dict
        assert "OverallDomain" in input_dict["Geometry"]
        assert "Fluids" in input_dict["Geometry"]
    return static_mix_type


def write_fvModel(input_dict, output_folder=".", force_sign=False):
    # Switch on the volumetric source: "ball" (new, exact-conservation
    # actuator-disk) vs "pancake" (legacy, default). The legacy path below is
    # left byte-for-byte unchanged.
    if input_dict.get(
        "volumetric_source", "pancake"
    ) == "ball" or input_dict.get("static_mixers"):
        write_fvModel_ball(input_dict, output_folder=output_folder)
        return
    mix_type = check_input(input_dict)
    write_preamble(output_folder)
    if "loop" in mix_type:
        geom_dict = from_block_rect_to_seg(input_dict["Geometry"])
        mesh_dict = input_dict["Meshing"]
    for imix, mtype in enumerate(mix_type):
        mixer = Mixer()
        if mtype == "expl":
            mixer.update_from_expl_dict(input_dict["mixers"][imix])
            if mixer.ready:
                if force_sign:
                    write_mixer_force_sign(mixer, output_folder)
                else:
                    write_mixer(mixer, output_folder)
        elif mtype == "loop":
            mixer.update_from_loop_dict(
                input_dict["mixers"][imix], geom_dict, mesh_dict
            )
            if mixer.ready:
                if force_sign:
                    write_mixer_force_sign(mixer, output_folder)
                else:
                    write_mixer(mixer, output_folder)

    write_end(output_folder)


def write_fvModel_ball(input_dict, output_folder="."):
    """Write the ``ball`` (actuator-disk) fvModels.

    Reads the top-level ``power`` (``from_P`` / ``from_Np_Vtip``) and
    ``momentum_source`` (``axial`` / ``axial_and_swirl``) modes; both default to
    the new full model. Each dynamic mixer is an
    :class:`~bird.preprocess.dynamic_mixer.mixer.ActuatorMixer`; each entry of the
    optional ``static_mixers`` list is a
    :class:`~bird.preprocess.dynamic_mixer.mixer.StaticMixer` appended to the same
    codedSource.
    """
    mix_type = check_input(input_dict)
    static_mix_type = check_static_input(input_dict)
    power_mode = input_dict.get("power", "from_Np_Vtip")
    momentum_mode = input_dict.get("momentum_source", "axial_and_swirl")
    write_preamble_ball(output_folder)
    if "loop" in mix_type or "loop" in static_mix_type:
        geom_dict = from_block_rect_to_seg(input_dict["Geometry"])
    for imix, mtype in enumerate(mix_type):
        mixer = ActuatorMixer()
        if mtype == "expl":
            mixer.update_from_expl_dict(input_dict["mixers"][imix])
        elif mtype == "loop":
            mixer.update_from_loop_dict(input_dict["mixers"][imix], geom_dict)
        if mixer.ready:
            write_mixer_ball(mixer, output_folder, power_mode, momentum_mode)
    for imix, mtype in enumerate(static_mix_type):
        mixer = StaticMixer()
        if mtype == "expl":
            mixer.update_from_expl_dict(input_dict["static_mixers"][imix])
        elif mtype == "loop":
            mixer.update_from_loop_dict(
                input_dict["static_mixers"][imix], geom_dict
            )
        if mixer.ready:
            write_static_mixer_ball(mixer, output_folder)
    write_end(output_folder)
