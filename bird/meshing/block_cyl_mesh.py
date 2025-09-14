import os

import numpy as np

from bird.meshing._mesh_tools import *
from bird.utilities.parser import parse_json, parse_yaml

from bird import logger


def assemble_geom(input_file, topo_file):
    if input_file.endswith(".yaml"):
        inpt_dict = parse_yaml(input_file)
    elif input_file.endswith(".json"):
        inpt_dict = parse_json(input_file)
    else:
        raise ValueError(f"unknown input file ({input_file}) extension")
    if topo_file.endswith(".yaml"):
        topo_dict = parse_yaml(topo_file)
    elif topo_file.endswith(".json"):
        topo_dict = parse_json(topo_file)
    else:
        raise ValueError(f"unknown topo file ({topo_file}) extension")

    # ~~~~ Define dimensions based on input
    r_dimensions_name = list(inpt_dict["Geometry"]["Radial"].keys())
    r_dimensions = [
        float(inpt_dict["Geometry"]["Radial"][dim])
        for dim in r_dimensions_name
    ]

    l_dimensions_name = list(inpt_dict["Geometry"]["Longitudinal"].keys())
    l_dimensions = [
        float(inpt_dict["Geometry"]["Longitudinal"][dim])
        for dim in l_dimensions_name
    ]

    # Order the dimensions of R and L
    r_coord = merge_and_sort(coord_list=r_dimensions, reverse_coord=False)
    l_coord = merge_and_sort(coord_list=l_dimensions, reverse_coord=True)
    dimension_dict = {"R": r_coord, "L": l_coord}

    # Define blocks that will be walls
    wall_dict = make_walls_from_topo(topo_dict)
    bound_dict = make_bound_from_topo(topo_dict)

    return {**wall_dict, **bound_dict, **dimension_dict}


def assemble_mesh(input_file, geom_dict):
    if input_file.endswith(".yaml"):
        inpt_dict = parse_yaml(input_file)
    elif input_file.endswith(".json"):
        inpt_dict = parse_json(input_file)
    else:
        raise ValueError(f"unknown input file ({input_file}) extension")
    R = geom_dict["R"]
    L = geom_dict["L"]
    N1 = len(R)
    N2 = len(L) - 1
    CW = []
    mCW = []
    C1 = []
    mC1 = []
    C2 = []
    mC2 = []
    for rval in R:
        CW.append(rval * 0.5)
        mCW.append(-rval * 0.5)
        C1.append(rval * np.cos(np.pi / 4))
        mC1.append(-rval * np.cos(np.pi / 4))
        C2.append(rval * np.sin(np.pi / 4))
        mC2.append(-rval * np.sin(np.pi / 4))

    NRSmallest = int(inpt_dict["Meshing"]["NRSmallest"])
    NVertSmallest = int(inpt_dict["Meshing"]["NVertSmallest"])
    NS = []
    NR = []
    NVert = []
    gradR_l = []
    gradR_r = []
    gradR = []
    gradVert = []

    # Radial meshing
    rad_len_block = np.zeros(len(R))
    rad_len_block[0] = R[0] / 2
    for i in range(len(R) - 1):
        rad_len_block[i + 1] = abs(R[i + 1] - R[i])

    try:
        iSmallest = int(inpt_dict["Meshing"]["iRSmallest"])
    except KeyError:
        iSmallest = np.argmin(rad_len_block)

    i_smallest_rad = iSmallest
    smallestRBlockSize = rad_len_block[i_smallest_rad]

    for irad_block in range(len(rad_len_block)):
        logger.debug(
            f"\tRadial block {irad_block+1} has size {rad_len_block[irad_block]}mm"
        )
    logger.debug(
        f"Smallest radial block {i_smallest_rad+1} with size {smallestRBlockSize}mm"
    )
    logger.debug(
        f"Radial mesh size in the smallest block will be {smallestRBlockSize}/{NRSmallest} = {smallestRBlockSize/NRSmallest:.2f}mm"
    )

    NR = [0 for i in range(len(R))]
    NR[iSmallest] = NRSmallest
    for i in range(len(R)):
        if not i == iSmallest:
            NR[i] = max(
                int(
                    round(
                        NRSmallest * abs(rad_len_block[i]) / smallestRBlockSize
                    )
                ),
                1,
            )
    NS = [NR[0] * 2]
    # Now figure out grading of each block
    for ir in range(len(R)):
        gradR_l.append(1.0)
        gradR_r.append(1.0)
        gradR.append(1.0)

    # Vertical meshing
    vert_len_block = np.array(
        [abs(L[i + 1] - L[i]) for i in range(len(L) - 1)]
    )

    try:
        iSmallest = int(inpt_dict["Meshing"]["iVertSmallest"])
    except KeyError:
        iSmallest = np.argmin(vert_len_block)

    i_smallest_vert = iSmallest
    smallestVertBlockSize = vert_len_block[i_smallest_vert]

    for ivert_block in range(len(vert_len_block)):
        logger.debug(
            f"\tVertical block {ivert_block+1} has size {vert_len_block[ivert_block]}mm"
        )
    logger.debug(
        f"Smallest vertical block {i_smallest_vert+1} with size {smallestVertBlockSize}mm"
    )
    logger.debug(
        f"Vertical mesh size in the smallest block will be {smallestVertBlockSize}/{NVertSmallest} = {smallestVertBlockSize/NVertSmallest:.2f}mm"
    )

    NVert = [0] * (len(L) - 1)
    NVert[iSmallest] = NVertSmallest
    for i in range(len(L) - 1):
        if not i == iSmallest:
            NVert[i] = max(
                int(
                    round(
                        NVert[iSmallest]
                        * abs(L[i + 1] - L[i])
                        / smallestVertBlockSize
                    )
                ),
                1,
            )
    for il in range(len(L) - 1):
        gradVert.append(1.0)

    # Mesh stretching
    try:
        verticalCoarseningProperties = inpt_dict["Meshing"][
            "verticalCoarsening"
        ]
        do_verticalCoarsening = True
    except KeyError:
        do_verticalCoarsening = False
    try:
        radialCoarseningProperties = inpt_dict["Meshing"]["radialCoarsening"]
        do_radialCoarsening = True
    except KeyError:
        do_radialCoarsening = False

    if do_verticalCoarsening:
        NVert, gradVert, minCellVert, maxCellVert = verticalCoarsening(
            ratio_properties=verticalCoarseningProperties,
            ref_block=i_smallest_vert,
            NVert=NVert,
            L=L,
            gradVert=gradVert,
            smooth=True,
        )
    else:
        block_length = [abs(L[i] - L[i + 1]) for i in range(len(NVert))]
        block_cell_length = [
            block_length[i] / NVert[i] for i in range(len(NVert))
        ]
        minCellVert = np.amin(block_cell_length)
        maxCellVert = np.amax(block_cell_length)
    if do_radialCoarsening:
        NR, gradR, minCellR, maxCellR = radialCoarsening(
            ratio_properties=radialCoarseningProperties,
            ref_block=i_smallest_rad,
            NR=NR,
            R=R,
            gradR=gradR,
            smooth=True,
        )
    else:
        block_length = [R[0] / 2] + [
            abs(R[i] - R[i + 1]) for i in range(len(R) - 1)
        ]
        block_cell_length = [block_length[i] / NR[i] for i in range(len(NR))]
        minCellR = np.amin(block_cell_length)
        maxCellR = np.amax(block_cell_length)

    logger.info("Vertical mesh:")
    logger.info(f"\tTotal NVert {sum(NVert)}")
    logger.info(f"\tNVert {NVert}")
    logger.info(f"\tsize min {minCellVert:.2f}mm max {maxCellVert:.2f}mm")
    logger.info("Radial mesh:")
    logger.info(f"\tTotal NR {sum(NR)}")
    logger.info(f"\tNR {NR}")
    logger.info(f"\tsize min {minCellR:.2f}mm max {maxCellR:.2f}mm")

    return {
        "NR": NR,
        "NS": NS,
        "NVert": NVert,
        "gradR": gradR,
        "gradVert": gradVert,
        "gradR_l": gradR_l,
        "gradR_r": gradR_r,
        "CW": CW,
        "mCW": mCW,
        "C1": C1,
        "mC1": mC1,
        "C2": C2,
        "mC2": mC2,
    }


def writeBlockMeshDict(out_folder, geom_dict, mesh_dict):
    outfile = os.path.join(out_folder, "blockMeshDict")

    R = geom_dict["R"]
    L = geom_dict["L"]
    r_wall = geom_dict["r_wall"]
    l_wall = geom_dict["l_wall"]
    BoundaryNames = geom_dict["names"]
    BoundaryType = geom_dict["types"]
    BoundaryRmin = geom_dict["rmin"]
    BoundaryRmax = geom_dict["rmax"]
    BoundaryLmin = geom_dict["lmin"]
    BoundaryLmax = geom_dict["lmax"]

    NR = mesh_dict["NR"]
    NS = mesh_dict["NS"]
    NVert = mesh_dict["NVert"]
    gradR = mesh_dict["gradR"]
    gradR_l = mesh_dict["gradR_l"]
    gradR_r = mesh_dict["gradR_r"]
    gradVert = mesh_dict["gradVert"]
    CW = mesh_dict["CW"]
    mCW = mesh_dict["mCW"]
    C1 = mesh_dict["C1"]
    mC1 = mesh_dict["mC1"]
    C2 = mesh_dict["C2"]
    mC2 = mesh_dict["mC2"]

    N1 = len(R)
    N2 = len(L) - 1

    # ~~~~ Write species Dict
    fw = open(outfile, "w+")
    # Write Header
    fw.write("FoamFile\n")
    fw.write("{\n")
    fw.write("    version     2.0;\n")
    fw.write("    format      ascii;\n")
    fw.write("    class       dictionary;\n")
    fw.write("    object      blockMeshDict;\n")
    fw.write("}\n")
    fw.write("\n")
    # fw.write('convertToMeters 0.001;\n')
    # Write all radii
    counter = 1
    for rval in R:
        fw.write("R" + str(counter) + " " + str(rval) + ";\n")
        counter = counter + 1
    fw.write("\n")
    # Write all minus radii
    counter = 1
    for rval in R:
        fw.write("mR" + str(counter) + " " + str(-rval) + ";\n")
        counter = counter + 1
    fw.write("\n")
    # Write all Length
    counter = 1
    for lval in L:
        fw.write("L" + str(counter) + " " + str(lval) + ";\n")
        counter = counter + 1
    fw.write("\n")
    # Write all C
    counter = 1
    for rval in R:
        fw.write("CW" + str(counter) + " " + str(CW[counter - 1]) + ";\n")
        fw.write("mCW" + str(counter) + " " + str(mCW[counter - 1]) + ";\n")
        fw.write("C1" + str(counter) + " " + str(C1[counter - 1]) + ";\n")
        fw.write("mC1" + str(counter) + " " + str(mC1[counter - 1]) + ";\n")
        fw.write("C2" + str(counter) + " " + str(C2[counter - 1]) + ";\n")
        fw.write("mC2" + str(counter) + " " + str(mC2[counter - 1]) + ";\n")
        fw.write("\n")
        counter = counter + 1

    # Write all Ngrid
    counter = 1
    for nR in NR:
        fw.write("NR" + str(counter) + " " + str(NR[counter - 1]) + ";\n")
        counter = counter + 1
    fw.write("\n")
    counter = 1
    for nS in NS:
        fw.write("NS" + str(counter) + " " + str(NS[counter - 1]) + ";\n")
        counter = counter + 1
    fw.write("\n")
    counter = 1
    for nVert in NVert:
        fw.write(
            "NVert" + str(counter) + " " + str(NVert[counter - 1]) + ";\n"
        )
        counter = counter + 1
    fw.write("\n")

    # ~~~~ Write vertices
    fw.write("vertices\n")
    fw.write("(\n")
    # Write the squares first
    counter = 0
    for i in range(len(L)):
        fw.write(f"     ($mCW1  $mCW1  $L{i+1})// {counter}\n")
        fw.write(f"     ( $CW1  $mCW1  $L{i+1})\n")
        fw.write(f"     ( $CW1   $CW1  $L{i+1})\n")
        fw.write(f"     ($mCW1   $CW1  $L{i+1})\n")
        fw.write("\n")
        counter = counter + 4
    # Write the circles then
    for ir in range(len(R)):
        for il in range(len(L)):
            fw.write(f"    ($mC1{ir+1}   $mC2{ir+1}   $L{il+1})// {counter}\n")
            fw.write(f"    ( $C1{ir+1}   $mC2{ir+1}   $L{il+1})\n")
            fw.write(f"    ( $C1{ir+1}   $C2{ir+1}    $L{il+1})\n")
            fw.write(f"    ($mC1{ir+1}   $C2{ir+1}    $L{il+1})\n")
            fw.write("\n")
            counter = counter + 4
    fw.write(");\n")
    fw.write("\n")

    # ~~~~ Write blocks
    fw.write("blocks\n")
    fw.write("(\n")
    # Write the squares first
    for i in range(N2):
        gradingVert = gradVert[i]
        # Am I a wall
        iwall = is_wall(l_wall, r_wall, -1, i)
        if iwall == 1:
            fw.write("//")

        i1 = int((i + 1) * 4)
        i2 = int(i * 4)
        fw.write(
            f"     hex ({i1} {i1+1} {i1+2} {i1+3} {i2} {i2+1} {i2+2} {i2+3})"
        )
        fw.write(
            f" ($NS1 $NS1 $NVert{i+1}) simpleGrading (1 1 {gradingVert})\n"
        )
    fw.write("\n")
    # Write the squares then
    for ir in range(N1):
        for il in range(N2):
            # gradingVert = 1
            # gradingR = 1
            # if il==N2-1:
            #    gradingVert = outletGrading
            gradingR_l = gradR_l[ir]
            gradingR_r = gradR_r[ir]
            gradingVert = gradVert[il]
            gradingR = gradR[ir]
            # Am I a wall
            iwall = is_wall(l_wall, r_wall, ir, il)
            # bottom right corner
            i1 = int(4 * (N2 + 1) * ir + 4 * (il + 1))
            # bottom left corner
            i2 = int(4 * (N2 + 1) * (ir + 1) + 4 * (il + 1))
            # top right corner
            i3 = int(4 * (N2 + 1) * ir + 4 * (il))
            # top left corner
            i4 = int(4 * (N2 + 1) * (ir + 1) + 4 * (il))
            # outlet
            if iwall == 1:
                fw.write("//")
            fw.write(
                f"     hex ({i1} {i2} {i2+1} {i1+1} {i3} {i4} {i4+1} {i3+1})"
            )
            fw.write(
                f" ($NR{ir+1} $NS1  $NVert{il+1}) simpleGrading ({gradingR} 1 {gradingVert})\n"
            )
            if iwall == 1:
                fw.write("//")
            fw.write(
                f"     hex ({i1+1} {i2+1} {i2+2} {i1+2} {i3+1} {i4+1} {i4+2} {i3+2})"
            )
            fw.write(
                f" ($NR{ir+1} $NS1  $NVert{il+1}) simpleGrading ({gradingR} 1 {gradingVert})\n"
            )
            if iwall == 1:
                fw.write("//")
            fw.write(
                f"     hex ({i1+2} {i2+2} {i2+3} {i1+3} {i3+2} {i4+2} {i4+3} {i3+3})"
            )
            fw.write(
                f" ($NR{ir+1} $NS1  $NVert{il+1}) simpleGrading ({gradingR} 1 {gradingVert})\n"
            )
            if iwall == 1:
                fw.write("//")
            fw.write(
                f"     hex ({i1+3} {i2+3} {i2} {i1} {i3+3} {i4+3} {i4} {i3})"
            )
            fw.write(
                f" ($NR{ir+1} $NS1  $NVert{il+1}) simpleGrading ({gradingR} 1 {gradingVert})\n"
            )
            fw.write("\n")
    fw.write(");\n")
    fw.write("\n")

    # ~~~~ Write edges
    fw.write("edges\n")
    fw.write("(\n")
    ind = len(L) * 4
    for ir in range(len(R)):
        for il in range(len(L)):
            # Edges should be removed if they are surrounded by walls
            iwall1 = is_wall(l_wall, r_wall, ir, il)
            iwall2 = is_wall(l_wall, r_wall, ir + 1, il)
            iwall3 = is_wall(l_wall, r_wall, ir, il - 1)
            iwall4 = is_wall(l_wall, r_wall, ir + 1, il - 1)
            sumwall = iwall1 + iwall2 + iwall3 + iwall4
            comment = 0
            if (
                sumwall == 4
                or (il == 0 and sumwall == 2)
                or (il == len(L) - 1 and sumwall == 2)
                or (ir == len(R) - 1 and sumwall == 2)
                or (ir == len(R) - 1 and il == 0 and sumwall == 1)
            ):
                comment = 1

            if comment == 1:
                fw.write("//")
            fw.write(f"    arc {ind} {ind+1} (0    $mR{ir+1} $L{il+1})\n")
            if comment == 1:
                fw.write("//")
            fw.write(f"    arc {ind+1} {ind+2} ($R{ir+1} 0 $L{il+1})\n")
            if comment == 1:
                fw.write("//")
            fw.write(f"    arc {ind+2} {ind+3} (0 $R{ir+1} $L{il+1})\n")
            if comment == 1:
                fw.write("//")
            fw.write(f"    arc {ind+3} {ind} ($mR{ir+1} 0 $L{il+1})\n")
            ind = ind + 4
            fw.write("\n")
    fw.write(");\n")
    fw.write("\n")

    # ~~~~ Write boundary
    fw.write("boundary\n")
    fw.write("(\n")

    for i, name in enumerate(BoundaryNames):
        fw.write(f"    {name}\n")
        fw.write("    {\n")
        if name.startswith("wall"):
            fw.write("        " + "type wall;\n")
        else:
            fw.write("        " + "type patch;\n")
        fw.write("        faces\n")
        fw.write("        (\n")

        for ibound in range(len(BoundaryType[i])):
            boundType = BoundaryType[i][ibound]
            if boundType == "lateral":
                rminInd = BoundaryRmin[i][ibound]
                rmaxInd = BoundaryRmax[i][ibound]
                lInd = BoundaryLmin[i][ibound]
                i1 = rminInd * (4 * (N2 + 1)) + 4 * lInd  # bottom
                i2 = i1 - 4  # top
                fw.write(f"            ( {i1} {i1+1} {i2+1} {i2})\n")
                fw.write(f"            ( {i1+1} {i1+2} {i2+2} {i2+1})\n")
                fw.write(f"            ( {i1+2} {i1+3} {i2+3} {i2+2})\n")
                fw.write(f"            ( {i1+3} {i1} {i2} {i2+3})\n")
                fw.write("\n")

            elif boundType == "top":
                lminInd = BoundaryLmin[i][ibound]
                lmaxInd = BoundaryLmax[i][ibound]
                rInd = BoundaryRmin[i][ibound]
                if rInd > 0:
                    i1 = 4 * (N2 + 1) * (rInd - 1) + 4 * lminInd  # right
                    i2 = i1 + 4 * (N2 + 1)  # left
                    fw.write(f"            ( {i1} {i2} {i2+1} {i1+1})\n")
                    fw.write(f"            ( {i1+1} {i2+1} {i1+2} {i2+2})\n")
                    fw.write(f"            ( {i1+2} {i2+2} {i2+3} {i1+3})\n")
                    fw.write(f"            ( {i1+3} {i2+3} {i2} {i1})\n")
                else:
                    i1 = lminInd * 4
                    fw.write(f"            ( {i1} {i1+1} {i1+2} {i1+3})\n")
                fw.write("\n")

            elif boundType == "bottom":
                lminInd = BoundaryLmin[i][ibound]
                lmaxInd = BoundaryLmax[i][ibound]
                rInd = BoundaryRmin[i][ibound]
                i1 = 4 * (N2 + 1) * (rInd - 1) + 4 * lminInd  # right
                i2 = i1 + 4 * (N2 + 1)  # left
                if rInd > 0:
                    fw.write(f"            ( {i1} {i2} {i2+1} {i1+1})\n")
                    fw.write(f"            ( {i1+1} {i2+1} {i1+2} {i2+2})\n")
                    fw.write(f"            ( {i1+2} {i2+2} {i2+3} {i1+3})\n")
                    fw.write(f"            ( {i1+3} {i2+3} {i2} {i1})\n")
                else:
                    i1 = lminInd * 4
                    fw.write(f"            ( {i1} {i1+1} {i1+2} {i1+3})\n")
                fw.write("\n")

        fw.write("        " + ");\n")
        fw.write("    " + "}\n")

    fw.write(");\n")
    fw.write("\n")

    # ~~~~ Write mergePatchPairs
    fw.write("mergePatchPairs\n")
    fw.write("(\n")
    fw.write(");\n")

    fw.write(
        "\n// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
    )
    fw.write("defaultPatch\n{type wall;}\n\n")

    fw.close()


def main(input_file, topo_file, output_folder):
    geom_dict = assemble_geom(input_file, topo_file)
    mesh_dict = assemble_mesh(input_file, geom_dict)
    writeBlockMeshDict(output_folder, geom_dict, mesh_dict)


if __name__ == "__main__":
    main()
