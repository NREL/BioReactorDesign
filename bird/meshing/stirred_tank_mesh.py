# from reactor_geom_data import *
import os

import numpy as np

from bird.meshing._stirred_tank_reactor import StirredTankReactor


def get_reactor_geom(yamlfile):
    return StirredTankReactor.from_file(yamlfile)


def write_ofoam_preamble(outfile, react):
    outfile.write(
        "/*--------------------------------*- C++ -*----------------------------------*\\\n"
    )
    outfile.write(
        "| =========                 |                                                 |\n"
    )
    outfile.write(
        "| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n"
    )
    outfile.write(
        "|  \\    /   O peration     | Version:  5                                     |\n"
    )
    outfile.write(
        "|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |\n"
    )
    outfile.write(
        "|    \\/     M anipulation  |                                                 |\n"
    )
    outfile.write(
        "\*---------------------------------------------------------------------------*/\n"
    )
    outfile.write("FoamFile\n")
    outfile.write("{\n")
    outfile.write("\tversion     2.0;\n")
    outfile.write("\tformat      ascii;\n")
    outfile.write("\tclass       dictionary;\n")
    outfile.write("\tobject      blockMeshDict;\n")
    outfile.write("}\n\n")
    outfile.write(
        "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
    )
    outfile.write("convertToMeters 1.0;\n\n")
    outfile.write(
        "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
    )


def write_vertices(outfile, react):
    outfile.write(
        "\n// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
    )
    outfile.write("vertices\n(\n")

    nsections = react.nsections
    reacthts = react.reacthts
    dangle = react.dangle
    nsplits = react.nsplits
    ncirc = react.ncirc
    circradii = react.circradii
    polyrad = react.polyrad

    counter = 0
    for repeat in range(2):
        for zi in range(nsections):
            outfile.write("\n//section " + str(zi) + "\n")
            outfile.write("\n//center\n")
            outfile.write(
                "(0.0 0.0 " + str(reacthts[zi]) + ") // " + str(counter) + "\n"
            )
            counter = counter + 1

            # polygon section
            outfile.write("\n//polygon\n")
            for i in range(nsplits):
                ang = i * dangle
                x = polyrad * np.cos(ang)
                y = polyrad * np.sin(ang)
                outfile.write(
                    "( "
                    + str(x)
                    + " "
                    + str(y)
                    + " "
                    + str(reacthts[zi])
                    + " ) // "
                    + str(counter)
                    + "\n"
                )
                counter = counter + 1

            outfile.write("\n//circles\n")

            for ci in range(ncirc):
                outfile.write("\n//circle " + str(ci) + "\n")

                for i in range(nsplits):
                    ang = i * dangle
                    x = circradii[ci] * np.cos(ang)
                    y = circradii[ci] * np.sin(ang)
                    outfile.write(
                        "( "
                        + str(x)
                        + " "
                        + str(y)
                        + " "
                        + str(reacthts[zi])
                        + " ) //"
                        + str(counter)
                        + "\n"
                    )
                    counter = counter + 1

    outfile.write(");\n")


def get_globalindex_of(splti, ci, zi, react):
    npts_per_section = react.npts_per_section
    centeroffset = react.centeroffset
    polyoffset = react.polyoffset
    nsplits = react.nsplits
    hub_circ = react.hub_circ
    tank_circ = react.tank_circ

    # also works for ci=-1
    global_id = (
        zi * npts_per_section
        + centeroffset
        + polyoffset
        + ci * nsplits
        + splti % nsplits
    )
    return global_id


def get_baffle_point_of(splti, ci, zi, react):
    baff_sections = react.baff_sections
    nsections = react.nsections
    npts_per_section = react.npts_per_section
    hub_circ = react.hub_circ
    tank_circ = react.tank_circ

    baffle_id = get_globalindex_of(splti, ci, zi, react)

    if zi in baff_sections:
        if ci == hub_circ:
            if splti % 2 == 0:  # even number for impeller
                baffle_id += nsections * npts_per_section

        if ci == tank_circ:
            if splti % 2 == 1:  # odd number for baffle
                baffle_id += nsections * npts_per_section

    return baffle_id


def write_edges(outfile, react):
    nsections = react.nsections
    nsplits = react.nsplits
    dangle = react.dangle
    circradii = react.circradii
    ncirc = react.ncirc
    reacthts = react.reacthts

    outfile.write(
        "\n// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
    )
    outfile.write("edges\n(\n")

    for zi in range(nsections):
        outfile.write("\n//section " + str(zi) + "\n")

        offset = 1 + nsplits  # one for center and nsplits for polygon

        outfile.write("\n//circles\n")
        for ci in range(ncirc):
            outfile.write("\n//circle " + str(ci) + "\n")
            for i in range(nsplits):
                ang = i * dangle
                midx = circradii[ci] * np.cos(ang + dangle / 2)
                midy = circradii[ci] * np.sin(ang + dangle / 2)

                globalind1 = get_baffle_point_of(i, ci, zi, react)
                globalind2 = get_globalindex_of(i + 1, ci, zi, react)

                outfile.write(
                    "arc " + str(globalind1) + " " + str(globalind2) + " "
                )
                outfile.write(
                    "( "
                    + str(midx)
                    + " "
                    + str(midy)
                    + " "
                    + str(reacthts[zi])
                    + " )\n"
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


def write_blocks(outfile, react):
    outfile.write(
        "\n// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
    )
    outfile.write("blocks\n(\n")

    idarray = np.zeros(8, dtype=int)
    mesharray = np.zeros(3, dtype=int)

    npts_per_section = react.npts_per_section
    nonstem_volumes = react.nonstem_volumes
    nsplits = react.nsplits
    centeroffset = react.centeroffset
    Na = react.Na
    Npoly = react.Npoly
    meshz = react.meshz
    meshr = react.meshr
    ncirc = react.ncirc
    inhub_circ = react.inhub_circ
    hub_circ = react.hub_circ
    mrf_circ = react.mrf_circ
    nvolumes = react.nvolumes
    mrf_volumes = react.mrf_volumes
    hub_volumes = react.hub_volumes

    for zi in range(nvolumes):
        outfile.write("\n//section " + str(zi) + "-" + str(zi + 1) + "\n")

        offset0 = zi * npts_per_section
        offset1 = (zi + 1) * npts_per_section

        # polygon section
        center_id0 = offset0
        center_id1 = offset1

        # skip polygon blocks in stem sections
        if zi in nonstem_volumes:
            for i in range(nsplits):
                localind1 = centeroffset + i % nsplits
                localind2 = centeroffset + (i + 1) % nsplits

                idarray[0] = offset0 + localind1
                idarray[1] = offset1 + localind1
                idarray[2] = offset1 + localind2
                idarray[3] = offset0 + localind2
                idarray[4] = center_id0
                idarray[5] = center_id1
                idarray[6] = center_id1
                idarray[7] = center_id0

                mesharray[0] = meshz[zi]
                mesharray[1] = Na
                mesharray[2] = Npoly

                zonename = "none"
                if zi in mrf_volumes:
                    zonename = "rotor"

                write_this_block(
                    outfile, "block %d" % (i), idarray, mesharray, zonename
                )

        idarray[:] = 0
        mesharray[:] = 0
        outfile.write("\n//circles\n")

        for ci in range(ncirc):
            zonename = "none"

            # skip blocks inside hub
            if ((ci == inhub_circ) or (ci == hub_circ)) and (
                zi in hub_volumes
            ):
                continue

            outfile.write("\n//circle " + str(ci) + "\n")

            if (zi in mrf_volumes) and (ci <= mrf_circ):
                zonename = "rotor"

            for i in range(nsplits):
                idarray[0] = get_baffle_point_of(i, ci, zi, react)
                idarray[1] = get_baffle_point_of(i, ci, zi + 1, react)
                idarray[2] = get_globalindex_of(i + 1, ci, zi + 1, react)
                idarray[3] = get_globalindex_of(i + 1, ci, zi, react)
                idarray[4] = get_baffle_point_of(i, ci - 1, zi, react)
                idarray[5] = get_baffle_point_of(i, ci - 1, zi + 1, react)
                idarray[6] = get_globalindex_of(i + 1, ci - 1, zi + 1, react)
                idarray[7] = get_globalindex_of(i + 1, ci - 1, zi, react)

                mesharray[0] = meshz[zi]
                mesharray[1] = Na
                mesharray[2] = meshr[ci]
                write_this_block(
                    outfile, "block %d" % (i), idarray, mesharray, zonename
                )

    outfile.write(");\n")

    # print "meshz:",meshz
    # print "meshr:",meshr


def write_patches(outfile, react):
    inhub_ci = react.inhub_circ
    hub_ci = react.hub_circ
    rot_ci = react.rot_circ
    npts_per_section = react.npts_per_section
    nsplits = react.nsplits
    hub_volumes = react.hub_volumes
    nsections = react.nsections
    ncirc = react.ncirc
    nimpellers = react.nimpellers
    baff_volumes = react.baff_volumes
    nonbaff_volumes = react.nonbaff_volumes
    only_stem_volumes = react.only_stem_volumes

    poly_ci = -1

    outfile.write(
        "\n// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
    )
    outfile.write("patches\n(\n")

    # inlet patch
    zi = 0
    centerid = zi * npts_per_section

    outfile.write("\n\tpatch inlet\n\t(\n")

    # polygon
    outfile.write("\n\t\t//polygon\n")
    for i in range(nsplits):
        outfile.write("\t\t( ")
        outfile.write(str(get_globalindex_of(i, poly_ci, zi, react)) + " ")
        outfile.write(str(get_globalindex_of(i + 1, poly_ci, zi, react)) + " ")
        outfile.write(str(centerid) + " ")
        outfile.write(str(centerid) + ")\n")

    outfile.write("\n\t\t//inhub_circ to polygon\n")
    for i in range(nsplits):
        outfile.write("\t\t( ")
        outfile.write(str(get_globalindex_of(i, inhub_ci, zi, react)) + " ")
        outfile.write(
            str(get_globalindex_of(i + 1, inhub_ci, zi, react)) + " "
        )
        outfile.write(str(get_globalindex_of(i + 1, poly_ci, zi, react)) + " ")
        outfile.write(str(get_globalindex_of(i, poly_ci, zi, react)) + ")\n")

    outfile.write("\n\t\t//hub to inhub_circ\n")
    for i in range(nsplits):
        outfile.write("\t\t( ")
        outfile.write(str(get_globalindex_of(i, hub_ci, zi, react)) + " ")
        outfile.write(str(get_globalindex_of(i + 1, hub_ci, zi, react)) + " ")
        outfile.write(
            str(get_globalindex_of(i + 1, inhub_ci, zi, react)) + " "
        )
        outfile.write(str(get_globalindex_of(i, inhub_ci, zi, react)) + ")\n")

    outfile.write("\n\t\t//rotor to hub\n")
    for i in range(nsplits):
        outfile.write("\t\t( ")
        outfile.write(str(get_globalindex_of(i, rot_ci, zi, react)) + " ")
        outfile.write(str(get_globalindex_of(i + 1, rot_ci, zi, react)) + " ")
        outfile.write(str(get_globalindex_of(i + 1, hub_ci, zi, react)) + " ")
        outfile.write(str(get_globalindex_of(i, hub_ci, zi, react)) + ")\n")

    outfile.write("\t)\n")

    # outlet patch
    zi = nsections - 1
    centerid = zi * npts_per_section
    outfile.write("\n\tpatch outlet\n\t(\n")

    # no polygon patch in outlet when we include stem
    # polygon
    # outfile.write("\n\t\t//polygon\n")
    # for i in range(nsplits):
    #    outfile.write("\t\t( ")
    #    outfile.write(str(get_globalindex_of(i,poly_ci,zi))+" ")
    #    outfile.write(str(get_globalindex_of(i+1,poly_ci,zi))+" ")
    #    outfile.write(str(centerid)+" ")
    #    outfile.write(str(centerid)+")\n")

    outfile.write("\n\t\t//circles\n")
    for ci in range(ncirc):
        outfile.write(
            "\n\t\t//circle " + str(ci) + " - " + str(ci - 1) + " \n"
        )
        for i in range(nsplits):
            outfile.write("\t\t( ")
            outfile.write(str(get_globalindex_of(i, ci, zi, react)) + " ")
            outfile.write(str(get_globalindex_of(i + 1, ci, zi, react)) + " ")
            outfile.write(
                str(get_globalindex_of(i + 1, ci - 1, zi, react)) + " "
            )
            outfile.write(
                str(get_globalindex_of(i, ci - 1, zi, react)) + ")\n"
            )

    outfile.write("\t)\n")

    # propeller patch
    outfile.write("\n\twall propeller\n\t(\n")

    # need polygon patch at the first impeller
    zi = hub_volumes[0]
    outfile.write("\n\t\t//polygon\n")
    centerid = zi * npts_per_section
    # polygon
    for i in range(nsplits):
        outfile.write("\t\t( ")
        outfile.write(str(get_globalindex_of(i, poly_ci, zi, react)) + " ")
        outfile.write(str(get_globalindex_of(i + 1, poly_ci, zi, react)) + " ")
        outfile.write(str(centerid) + " ")
        outfile.write(str(centerid) + ")\n")

    for n_imp in range(nimpellers):
        zi_bottom = hub_volumes[n_imp]  # bottom of impeller section
        zi_top = zi_bottom + 1  # bottom of impeller section

        for zi in [zi_bottom, zi_top]:
            outfile.write("\n\t\t//hub to blade circle\n")
            for i in range(nsplits):
                outfile.write("\t\t( ")
                outfile.write(
                    str(get_baffle_point_of(i, hub_ci, zi, react)) + " "
                )
                outfile.write(
                    str(get_globalindex_of(i + 1, hub_ci, zi, react)) + " "
                )
                outfile.write(
                    str(get_globalindex_of(i + 1, inhub_ci, zi, react)) + " "
                )
                outfile.write(
                    str(get_baffle_point_of(i, inhub_ci, zi, react)) + ")\n"
                )

            outfile.write("\n\t\t//blade to polygon\n")
            for i in range(nsplits):
                outfile.write("\t\t( ")
                outfile.write(
                    str(get_baffle_point_of(i, inhub_ci, zi, react)) + " "
                )
                outfile.write(
                    str(get_globalindex_of(i + 1, inhub_ci, zi, react)) + " "
                )
                outfile.write(
                    str(get_globalindex_of(i + 1, poly_ci, zi, react)) + " "
                )
                outfile.write(
                    str(get_baffle_point_of(i, poly_ci, zi, react)) + ")\n"
                )

        # sides
        outfile.write("\n\t\t//sides\n")
        for i in range(nsplits):
            outfile.write("\t\t( ")
            outfile.write(
                str(get_baffle_point_of(i, hub_ci, zi_bottom, react)) + " "
            )
            outfile.write(
                str(get_globalindex_of(i + 1, hub_ci, zi_bottom, react)) + " "
            )
            outfile.write(
                str(get_globalindex_of(i + 1, hub_ci, zi_top, react)) + " "
            )
            outfile.write(
                str(get_baffle_point_of(i, hub_ci, zi_top, react)) + ")\n"
            )

    # blades
    outfile.write("\n\t\t//blades\n")
    for zi in baff_volumes:
        zi_bottom = zi
        zi_top = zi + 1
        for i in range(0, nsplits, 2):  # even numbers
            outfile.write("\t\t( ")
            outfile.write(
                str(get_baffle_point_of(i, hub_ci + 1, zi_bottom, react)) + " "
            )
            outfile.write(
                str(get_baffle_point_of(i, hub_ci + 1, zi_top, react)) + " "
            )
            outfile.write(
                str(get_baffle_point_of(i, hub_ci, zi_top, react)) + " "
            )
            outfile.write(
                str(get_baffle_point_of(i, hub_ci, zi_bottom, react)) + ")\n"
            )

            outfile.write("\t\t( ")
            outfile.write(
                str(get_globalindex_of(i, hub_ci + 1, zi_bottom, react)) + " "
            )
            outfile.write(
                str(get_globalindex_of(i, hub_ci + 1, zi_top, react)) + " "
            )
            outfile.write(
                str(get_globalindex_of(i, hub_ci, zi_top, react)) + " "
            )
            outfile.write(
                str(get_globalindex_of(i, hub_ci, zi_bottom, react)) + ")\n"
            )

    # inside hub blades
    for n_imp in range(nimpellers):
        hub_vol = hub_volumes[n_imp]
        zi_pairs = [[hub_vol - 1, hub_vol], [hub_vol + 1, hub_vol + 2]]

        for zi_pair in zi_pairs:
            zi_below = zi_pair[0]
            zi_above = zi_pair[1]

            for i in range(0, nsplits, 2):  # even numbers
                outfile.write("\t\t( ")
                outfile.write(
                    str(get_baffle_point_of(i, hub_ci, zi_below, react)) + " "
                )
                outfile.write(
                    str(get_baffle_point_of(i, inhub_ci, zi_below, react))
                    + " "
                )
                outfile.write(
                    str(get_baffle_point_of(i, inhub_ci, zi_above, react))
                    + " "
                )
                outfile.write(
                    str(get_baffle_point_of(i, hub_ci, zi_above, react))
                    + ")\n"
                )

                outfile.write("\t\t( ")
                outfile.write(
                    str(get_globalindex_of(i, hub_ci, zi_below, react)) + " "
                )
                outfile.write(
                    str(get_globalindex_of(i, inhub_ci, zi_below, react)) + " "
                )
                outfile.write(
                    str(get_globalindex_of(i, inhub_ci, zi_above, react)) + " "
                )
                outfile.write(
                    str(get_globalindex_of(i, hub_ci, zi_above, react)) + ")\n"
                )

    # stem
    outfile.write("\n\t\t//stem sides\n")
    for zi in only_stem_volumes:
        zi_bottom = zi
        zi_top = zi + 1
        for i in range(nsplits):
            outfile.write("\t\t( ")
            outfile.write(
                str(get_globalindex_of(i, poly_ci, zi_bottom, react)) + " "
            )
            outfile.write(
                str(get_globalindex_of(i + 1, poly_ci, zi_bottom, react)) + " "
            )
            outfile.write(
                str(get_globalindex_of(i + 1, poly_ci, zi_top, react)) + " "
            )
            outfile.write(
                str(get_globalindex_of(i, poly_ci, zi_top, react)) + ")\n"
            )

    outfile.write("\t)\n")

    # stator and walls patch
    tank_ci = ncirc - 1
    outfile.write("\n\twall walls\n\t(\n")

    for zi in range(nsections - 1):
        outfile.write(
            "\n\t\t//tank walls " + str(zi) + " - " + str(zi + 1) + "\n"
        )

        for i in range(nsplits):
            outfile.write("\t\t( ")
            outfile.write(
                str(get_baffle_point_of(i, tank_ci, zi, react)) + " "
            )
            outfile.write(
                str(get_globalindex_of(i + 1, tank_ci, zi, react)) + " "
            )
            outfile.write(
                str(get_globalindex_of(i + 1, tank_ci, zi + 1, react)) + " "
            )
            outfile.write(
                str(get_baffle_point_of(i, tank_ci, zi + 1, react)) + ")\n"
            )

        # baffles
        outfile.write("\n\t\t//baffles\n")
        for i in range(1, nsplits, 2):  # all odd numbers
            outfile.write("\t\t( ")
            outfile.write(
                str(get_baffle_point_of(i, tank_ci, zi, react)) + " "
            )
            outfile.write(
                str(get_baffle_point_of(i, tank_ci - 1, zi, react)) + " "
            )
            outfile.write(
                str(get_baffle_point_of(i, tank_ci - 1, zi + 1, react)) + " "
            )
            outfile.write(
                str(get_baffle_point_of(i, tank_ci, zi + 1, react)) + ")\n"
            )

            outfile.write("\t\t( ")
            outfile.write(str(get_globalindex_of(i, tank_ci, zi, react)) + " ")
            outfile.write(
                str(get_globalindex_of(i, tank_ci - 1, zi, react)) + " "
            )
            outfile.write(
                str(get_globalindex_of(i, tank_ci - 1, zi + 1, react)) + " "
            )
            outfile.write(
                str(get_globalindex_of(i, tank_ci, zi + 1, react)) + ")\n"
            )

    # inlet wall patch
    # skip rotor to hub to inhub_circ to polygon region
    # which is covered in inflow
    zi = 0
    outfile.write("\n\t\t//circles\n")
    for ci in range(rot_ci + 1, ncirc):  # start from rotor circle
        outfile.write(
            "\n\t\t//circle " + str(ci) + " - " + str(ci - 1) + " \n"
        )
        for i in range(nsplits):
            outfile.write("\t\t( ")
            outfile.write(str(get_globalindex_of(i, ci, zi, react)) + " ")
            outfile.write(str(get_globalindex_of(i + 1, ci, zi, react)) + " ")
            outfile.write(
                str(get_globalindex_of(i + 1, ci - 1, zi, react)) + " "
            )
            outfile.write(
                str(get_globalindex_of(i, ci - 1, zi, react)) + ")\n"
            )

    outfile.write("\t)\n")

    # stitch faces patches
    outfile.write("\n\tempty inside_to_hub\n\t(\n")

    zi_pairs = []
    for vols in nonbaff_volumes:
        zi_pairs.append([vols, vols + 1])

    for zi_pair in zi_pairs:
        zi_below = zi_pair[0]
        zi_above = zi_pair[1]
        outfile.write(
            "\n\t\t//pair :" + str(zi_below) + "-" + str(zi_above) + "\n"
        )

        for i in range(0, nsplits, 2):  # even numbers
            outfile.write("\t\t( ")
            outfile.write(
                str(get_baffle_point_of(i, hub_ci, zi_below, react)) + " "
            )
            outfile.write(
                str(get_baffle_point_of(i, inhub_ci, zi_below, react)) + " "
            )
            outfile.write(
                str(get_baffle_point_of(i, inhub_ci, zi_above, react)) + " "
            )
            outfile.write(
                str(get_baffle_point_of(i, hub_ci, zi_above, react)) + ")\n"
            )

    outfile.write("\t)\n")

    outfile.write("\n\tempty inside_to_hub_copy\n\t(\n")

    for zi_pair in zi_pairs:
        zi_below = zi_pair[0]
        zi_above = zi_pair[1]
        outfile.write(
            "\n\t\t//pair :" + str(zi_below) + "-" + str(zi_above) + "\n"
        )

        for i in range(0, nsplits, 2):  # even numbers
            outfile.write("\t\t( ")
            outfile.write(
                str(get_globalindex_of(i, hub_ci, zi_below, react)) + " "
            )
            outfile.write(
                str(get_globalindex_of(i, inhub_ci, zi_below, react)) + " "
            )
            outfile.write(
                str(get_globalindex_of(i, inhub_ci, zi_above, react)) + " "
            )
            outfile.write(
                str(get_globalindex_of(i, hub_ci, zi_above, react)) + ")\n"
            )

    outfile.write("\t)\n")

    # stitch faces patches
    outfile.write("\n\tempty hub_to_rotor\n\t(\n")

    for zi_pair in zi_pairs:
        zi_below = zi_pair[0]
        zi_above = zi_pair[1]
        outfile.write(
            "\n\t\t//pair :" + str(zi_below) + "-" + str(zi_above) + "\n"
        )

        for i in range(0, nsplits, 2):  # even numbers
            outfile.write("\t\t( ")
            outfile.write(
                str(get_baffle_point_of(i, rot_ci, zi_below, react)) + " "
            )
            outfile.write(
                str(get_baffle_point_of(i, hub_ci, zi_below, react)) + " "
            )
            outfile.write(
                str(get_baffle_point_of(i, hub_ci, zi_above, react)) + " "
            )
            outfile.write(
                str(get_baffle_point_of(i, rot_ci, zi_above, react)) + ")\n"
            )

    outfile.write("\t)\n")
    outfile.write("\n\tempty hub_to_rotor_copy\n\t(\n")

    for zi_pair in zi_pairs:
        zi_below = zi_pair[0]
        zi_above = zi_pair[1]
        outfile.write(
            "\n\t\t//pair :" + str(zi_below) + "-" + str(zi_above) + "\n"
        )

        for i in range(0, nsplits, 2):  # even numbers
            outfile.write("\t\t( ")
            outfile.write(
                str(get_globalindex_of(i, rot_ci, zi_below, react)) + " "
            )
            outfile.write(
                str(get_globalindex_of(i, hub_ci, zi_below, react)) + " "
            )
            outfile.write(
                str(get_globalindex_of(i, hub_ci, zi_above, react)) + " "
            )
            outfile.write(
                str(get_globalindex_of(i, rot_ci, zi_above, react)) + ")\n"
            )

    outfile.write("\t)\n")

    outfile.write(");\n")


if __name__ == "__main__":
    react = get_reactor_geom(
        "stirred_tank_mesh_templates/base_tank/tank_par.yaml"
    )
    with open("tmp", "w") as outfile:
        write_ofoam_preamble(outfile, react)
        write_vertices(outfile, react)
        write_edges(outfile, react)
        write_blocks(outfile, react)
        write_patches(outfile, react)
