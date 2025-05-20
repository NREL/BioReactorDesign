import numpy as np

from bird.meshing._mesh_tools import parseJsonFile
from bird.meshing.block_rect_mesh import from_block_rect_to_seg
from bird.preprocess.dynamic_mixer.io_fvModels import *
from bird.preprocess.dynamic_mixer.mixer import Mixer


def check_input(input_dict):
    assert isinstance(input_dict, dict)
    mix_type = []
    for mix in input_dict["mixers"]:
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


def write_fvModel(input_dict, output_folder=".", force_sign=False):
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


if __name__ == "__main__":
    input_dict = parseJsonFile(
        os.path.join("mixing_template", "loop_reactor_list", "mixers.json"),
    )
    write_fvModel(input_dict)
