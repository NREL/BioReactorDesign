import os
import sys

import numpy as np

from bird.meshing._mesh_tools import *


def write_ofoam_preamble(outfile):
    outfile.write("FoamFile\n")
    outfile.write("{\n")
    outfile.write("    version     2.0;\n")
    outfile.write("    format      ascii;\n")
    outfile.write("    class       dictionary;\n")
    outfile.write("    object      blockMeshDict;\n")
    outfile.write("}\n")
    outfile.write("\n")
    outfile.write("convertToMeters 1.0;\n\n")


def write_vertices(outfile, lengths, nboxes):
    # nx,ny, nz are boxes
    delta = lengths[:] / nboxes[:]

    outfile.write(
        "\n// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
    )
    outfile.write("vertices\n(\n")
    for k in range(nboxes[2] + 1):
        for j in range(nboxes[1] + 1):
            for i in range(nboxes[0] + 1):
                outfile.write(
                    "( "
                    + str(i * delta[0])
                    + " "
                    + str(j * delta[1])
                    + "  "
                    + str(k * delta[2])
                    + ")\n"
                )
    outfile.write(");\n")


def write_this_block(outfile, comment, ids, mesh, zonename="none"):
    outfile.write("\n //" + comment + "\n")
    outfile.write("hex (")
    for i in range(len(ids)):
        outfile.write(str(ids[i]) + " ")
    outfile.write(")\n")

    if zonename != "none":
        outfile.write(zonename + "\n")

    outfile.write("( %d %d %d )\n" % (mesh[0], mesh[1], mesh[2]))
    outfile.write("SimpleGrading (1 1 1)\n")


def write_blocks(outfile, blockids, lengths, nboxes, points_per_len):
    nx = nboxes[0]
    ny = nboxes[1]
    nz = nboxes[2]

    idarray = np.zeros(8, dtype=int)
    mesharray = np.zeros(3, dtype=int)

    mesharray[:] = np.ceil(lengths[:] * points_per_len[:] / nboxes[:])

    nblocks = len(blockids)
    outfile.write(
        "\n// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
    )
    outfile.write("blocks\n(\n")

    for bl in range(nblocks):
        i = blockids[bl][0]
        j = blockids[bl][1]
        k = blockids[bl][2]

        for kk in range(2):
            for jj in range(2):
                for ii in range(2):
                    ri = ii
                    if jj == 1:
                        ri = 1 - ii  # reverse i
                    ind = (
                        (k + kk) * (nx + 1) * (ny + 1)
                        + (j + jj) * (nx + 1)
                        + (i + ri)
                    )
                    idarray[kk * 4 + jj * 2 + ii] = ind

        write_this_block(outfile, "block %d" % (bl), idarray, mesharray)

    outfile.write(");\n")


def write_patches(outfile):
    outfile.write(
        "\n// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
    )
    outfile.write("defaultPatch\n{   type wall;}\n\n")
    outfile.write("patches\n(\n")
    outfile.write(");\n")


def make_fluid_blocks_from_corner(corners):
    assert isinstance(corners, list)
    fluid_blocks = []
    for multi_block in corners:
        n_end_block = len(multi_block)
        for iend, end_block in enumerate(multi_block):
            # Add blocks that connect end points
            if iend == 0:
                # Make sure to not add endPoint if already there
                if list(end_block) not in fluid_blocks:
                    fluid_blocks.append(list(end_block))
            else:
                # Find direction to previous end point
                prev_block = multi_block[iend - 1]
                line = np.array(end_block, dtype=int) - np.array(
                    prev_block, dtype=int
                )
                nnz = np.nonzero(line)
                assert len(nnz) == 1
                direction = np.array(line / np.linalg.norm(line), dtype=int)
                blocks_line = []
                found_next = False
                count = 0
                # Add blocks until next end point is found
                while not found_next:
                    count += int(1)
                    new_block = (
                        np.array(prev_block, dtype=int) + count * direction
                    )
                    if (
                        np.linalg.norm(
                            new_block - np.array(end_block, dtype=int)
                        )
                        < 1
                    ):
                        found_next = True
                    blocks_line.append(
                        list(
                            np.array(prev_block, dtype=int) + count * direction
                        )
                    )
                fluid_blocks += blocks_line

    return fluid_blocks


def assemble_geom(input_file):
    # inpt = parseJsonFile(input_file)
    if input_file.endswith(".yaml"):
        inpt = parseYAMLFile(input_file)
    elif input_file.endswith(".json"):
        inpt = parseJsonFile(input_file)
    else:
        raise ValueError(f"unknown input file ({input_file}) extension")

    # ~~~~ Define background domain
    domain = inpt["Geometry"]["OverallDomain"]
    overall_nblocks = np.array(
        [
            domain["x"]["nblocks"],
            domain["y"]["nblocks"],
            domain["z"]["nblocks"],
        ],
        dtype=int,
    )
    overall_size_per_block = np.array(
        [
            domain["x"]["size_per_block"],
            domain["y"]["size_per_block"],
            domain["z"]["size_per_block"],
        ]
    )
    overall_size = overall_nblocks * overall_size_per_block

    # ~~~~ Define corners
    corners = [entry for entry in inpt["Geometry"]["Fluids"]]
    fluid_blocks = make_fluid_blocks_from_corner(corners)

    return {
        "overall_size": overall_size,
        "overall_nblocks": overall_nblocks,
        "fluid_blocks": fluid_blocks,
    }


def from_block_rect_to_seg(input_geom_dict, rescale=True):
    blocksize_x = input_geom_dict["OverallDomain"]["x"]["size_per_block"]
    if rescale:
        blocksize_x *= input_geom_dict["OverallDomain"]["x"]["rescale"]
    blocksize_y = input_geom_dict["OverallDomain"]["y"]["size_per_block"]
    if rescale:
        blocksize_y *= input_geom_dict["OverallDomain"]["y"]["rescale"]
    blocksize_z = input_geom_dict["OverallDomain"]["z"]["size_per_block"]
    if rescale:
        blocksize_z *= input_geom_dict["OverallDomain"]["z"]["rescale"]
    segments = {}
    iseg = 0
    for ifl, fluid_list in enumerate(input_geom_dict["Fluids"]):
        nblock = len(fluid_list)
        # if ifl > 0:
        #    segments[iseg] = {}
        #    segments[iseg]["blocks"] = [
        #        segments[iseg - 1]["blocks"][-1],
        #        fluid_list[i],
        #    ]
        #    iseg += 1
        for i in range(nblock - 1):
            segments[iseg] = {}
            segments[iseg]["blocks"] = [fluid_list[i], fluid_list[i + 1]]
            iseg += 1
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

    return {
        "segments": segments,
        "blocksize": [blocksize_x, blocksize_y, blocksize_z],
    }


def assemble_mesh(input_file):
    if input_file.endswith(".yaml"):
        inpt = parseYAMLFile(input_file)
    elif input_file.endswith(".json"):
        inpt = parseJsonFile(input_file)
    else:
        raise ValueError(f"unknown input file ({input_file}) extension")

    # ~~~~ Define num points per block
    block_points = inpt["Meshing"]["Blockwise"]
    pperlen = np.array(
        [block_points["x"], block_points["y"], block_points["z"]], dtype=int
    )

    return {
        "pperlen": pperlen,
    }


def writeBlockMeshDict(out_folder, geomDict, meshDict):
    n_blocks = len(geomDict["fluid_blocks"])
    blockids = np.array(geomDict["fluid_blocks"]).reshape(n_blocks, 3)

    with open(os.path.join(out_folder, "blockMeshDict"), "w+") as outfile:
        write_ofoam_preamble(outfile)
        write_vertices(
            outfile, geomDict["overall_size"], geomDict["overall_nblocks"]
        )
        write_blocks(
            outfile,
            blockids,
            geomDict["overall_size"],
            geomDict["overall_nblocks"],
            meshDict["pperlen"],
        )
        write_patches(outfile)


def main(input_file, output_folder):
    geomDict = assemble_geom(input_file)
    meshDict = assemble_mesh(input_file)
    writeBlockMeshDict(output_folder, geomDict, meshDict)
