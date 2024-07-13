import json
import sys

import numpy as np
import stl

from bird.meshing._mesh_tools import parseJsonFile
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
    return mix_type


def proc_geom(input_geom_dict):
    blocksize_x = (
        input_geom_dict["OverallDomain"]["x"]["size_per_block"]
        * input_geom_dict["OverallDomain"]["x"]["rescale"]
    )
    blocksize_y = (
        input_geom_dict["OverallDomain"]["y"]["size_per_block"]
        * input_geom_dict["OverallDomain"]["y"]["rescale"]
    )
    blocksize_z = (
        input_geom_dict["OverallDomain"]["z"]["size_per_block"]
        * input_geom_dict["OverallDomain"]["z"]["rescale"]
    )
    segments = {}
    isegment = 0
    for fluid_list in input_geom_dict["Fluids"]:
        nblock = len(fluid_list)
        for i in range(nblock - 1):
            segments[isegment] = {}
            segments[isegment]["blocks"] = [fluid_list[i], fluid_list[i + 1]]
            isegment += 1
    for iseg in segments:
        segments[iseg]["start"] = np.array(
            [
                blocksize_x * segments[iseg]["blocks"][0][0]
                + blocksize_x * 0.5,
                blocksize_y * segments[iseg]["blocks"][0][1]
                + blocksize_y * 0.5,
                blocksize_z * segments[iseg]["blocks"][0][2]
                + blocksize_z * 0.5,
            ]
        )
        segments[iseg]["end"] = np.array(
            [
                blocksize_x * segments[iseg]["blocks"][1][0]
                + blocksize_x * 0.5,
                blocksize_y * segments[iseg]["blocks"][1][1]
                + blocksize_y * 0.5,
                blocksize_z * segments[iseg]["blocks"][1][2]
                + blocksize_z * 0.5,
            ]
        )
        vec_conn = segments[iseg]["end"] - segments[iseg]["start"]
        segments[iseg]["conn"] = vec_conn
        norm_vec_conn = np.linalg.norm(vec_conn)
        segments[iseg]["normal_dir"] = int(np.nonzero(vec_conn)[0][0])
        if segments[iseg]["normal_dir"] == 0:
            segments[iseg]["max_rad"] = (blocksize_y + blocksize_z) / 4
        if segments[iseg]["normal_dir"] == 1:
            segments[iseg]["max_rad"] = (blocksize_x + blocksize_z) / 4
        if segments[iseg]["normal_dir"] == 2:
            segments[iseg]["max_rad"] = (blocksize_x + blocksize_y) / 4

    return {"segments": segments}


def write_fvModel(input_dict, output_folder="."):
    mix_type = check_input(input_dict)
    write_preamble(output_folder)
    if "loop" in mix_type:
        geom_dict = proc_geom(input_dict["Geometry"])
    for imix, mtype in enumerate(mix_type):
        mixer = Mixer()
        if mtype == "expl":
            mixer.update_from_expl_dict(input_dict["mixers"][imix])
            if mixer.ready:
                write_mixer(mixer, output_folder)
        elif mtype == "loop":
            mixer.update_from_loop_dict(input_dict["mixers"][imix], geom_dict)
            if mixer.ready:
                write_mixer(mixer, output_folder)

    write_end(output_folder)


if __name__ == "__main__":
    input_dict = parseJsonFile(
        os.path.join("mixing_template", "loop_reactor_list", "mixers.json"),
    )
    write_fvModel(input_dict)
