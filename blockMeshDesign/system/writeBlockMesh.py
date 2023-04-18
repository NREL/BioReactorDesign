import numpy as np
import os
import sys


def fun(G, N1):
    result = (1.0 - G) / (G * (1 - np.power(G, 1.0 / N1)))
    return result


# def fun(G,N1):
#    result = (1.0-G**(N1/(N1-1)))/((G**(1.0+1.0/(N1-1)))*(1-np.power(G,1.0/(N1-1))))
#    return result


def bissection(val, fun, N1):
    Gmin = 0.01
    Gmax = 100
    resultmin = fun(Gmin, N1) - val
    resultmax = fun(Gmax, N1) - val
    if resultmin * resultmax > 0:
        print("Error,the initial bounds of grading do not encompass the solution")
        # stop

    for i in range(100):
        Gmid = 0.5 * (Gmax + Gmin)
        resultmid = fun(Gmid, N1) - val
        if resultmid * resultmax < 0:
            Gmin = Gmid
            resultmin = resultmid
        else:
            Gmax = Gmid
            resultmax = resultmid

    return Gmid


def amIwall(WallL, WallR, ir, il):
    # returns 1 if block is wall
    # returns 0 otherwise
    iwall = 0
    for iw in range(len(WallL)):
        if WallL[iw] == (il + 1) and WallR[iw] == ir + 1:
            iwall = 1
    return iwall


def mergeSort(list, reverse):
    list.sort(reverse=reverse)
    listtmp = []
    for val in list:
        if len(listtmp) == 0 or not val == listtmp[-1]:
            listtmp.append(val)
    return listtmp


def verticalOutletCoarsening(ratio, NVert, gradVert=None, smooth=False):
    if ratio > 1:
        sys.exit("ERROR: vertical coarsening ratio should be < 1")

    NVert[0] = int(NVert[0] * ratio)

    if smooth:
        if gradVert is None:
            sys.exit("Error: cannot smooth vertical transition without grading list")

        Length = L[0] - L[1]
        deltaE = (L[1] - L[2]) / NVert[1]
        gradVert[0] = 1 / (bissection(Length / deltaE, fun, NVert[0]))

    return NVert, gradVert


def radialFlowCoarseing(ratio, NR, gradR=None, smooth=False):
    if ratio > 1:
        sys.exit("ERROR: radial coarsening ratio should be < 1")

    NR[2] = int(NR[2] * ratio)

    if smooth:
        if gradR is None:
            sys.exit("ERROR: cannot smooth radial transition without grading list")

        Length = R[2] - R[1]
        deltaE = (R[1] - R[0]) / NR[1]
        gradR[2] = 1 / (bissection(Length / deltaE, fun, NR[2]))

    return NR, gradR


# ~~~ Initialize input
input_file = sys.argv[1]

# ~~~~ Parse input
inpt = {}
f = open(input_file)
data = f.readlines()
for line in data:
    if ":" in line:
        key, value = line.split(":")
        inpt[key.strip()] = value.strip()
f.close()

# ~~~~ Define dimensions based on input
RSupport = float(inpt["RSupport"])
RSparger = float(inpt["RSparger"])
RColumn = float(inpt["RColumn"])

LColumnTop = float(inpt["LColumnTop"])
LSpargerTop = float(inpt["LSpargerTop"])
LSupportTop = float(inpt["LSupportTop"])
LBottom = float(inpt["LBottom"])

outfile = "blockMeshDict"


# Dimensions
R = [RSupport, RSparger, RColumn]
L = [LColumnTop, LSpargerTop, LSupportTop, LBottom]

# Merge and sort R and L
R = mergeSort(R, False)
L = mergeSort(L, True)

# Define blocks that will be walls
WallR = []
WallL = []
# Support
WallR.append(0)
WallL.append(3)
WallR.append(1)
WallL.append(3)
# Sparger
WallR.append(0)
WallL.append(2)
WallR.append(1)
WallL.append(2)
WallR.append(2)
WallL.append(2)

# Define boundaries
BoundaryNames = []
BoundaryType = []
BoundaryRmin = []
BoundaryRmax = []
BoundaryLmin = []
BoundaryLmax = []
BoundaryNames.append("Support")
BoundaryType.append(["lateral"])
BoundaryRmin.append([1])
BoundaryRmax.append([2])
BoundaryLmin.append([3])
BoundaryLmax.append([3])
BoundaryNames.append("SpargerWalls")
BoundaryType.append(["bottom", "top", "top", "top"])
BoundaryRmin.append([2, 0, 1, 2])
BoundaryRmax.append([2, 0, 1, 2])
BoundaryLmin.append([2, 1, 1, 1])
BoundaryLmax.append([3, 2, 2, 2])
BoundaryNames.append("SpargerInflow")
BoundaryType.append(["lateral"])
BoundaryRmin.append([2])
BoundaryRmax.append([3])
BoundaryLmin.append([2])
BoundaryLmax.append([2])
BoundaryNames.append("BCRInflow")
BoundaryType.append(["bottom", "bottom"])
BoundaryRmin.append([2, 3])
BoundaryRmax.append([2, 3])
BoundaryLmin.append([3, 3])
BoundaryLmax.append([4, 4])
BoundaryNames.append("BCROutflow")
BoundaryType.append(["top", "top", "top", "top"])
BoundaryRmin.append([0, 1, 2, 3])
BoundaryRmax.append([0, 1, 2, 3])
BoundaryLmin.append([0, 0, 0, 0])
BoundaryLmax.append([1, 1, 1, 1])
BoundaryNames.append("BCRWalls")
BoundaryType.append(["lateral", "lateral", "lateral"])
BoundaryRmin.append([3, 3, 3])
BoundaryRmax.append([3, 3, 3])
BoundaryLmin.append([1, 2, 3])
BoundaryLmax.append([1, 2, 3])

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

NS_in = int(inpt["NS"])
NVert_topBlock = int(inpt["NVert_topBlock"])
NS = []
NR = []
NVert = []
gradR_l = []
gradR_r = []
gradR = []
gradVert = []


# Radial meshing
NS.append(NS_in)
NR.append(int(NS[0] / 2))
# Uniform meshing
for i in range(len(R) - 1):
    NR.append(max(int(round(NR[0] * abs(R[i + 1] - R[i]) / abs(R[0] - R[0] / 2))), 1))
# Now figure out grading of each block
for ir in range(len(R)):
    gradR_l.append(1.0)
    gradR_r.append(1.0)
    gradR.append(1.0)

# Vertical meshing
NVert.append(NVert_topBlock)
for i in range(len(L) - 2):
    NVert.append(
        max(int(round(NVert[0] * abs(L[i + 2] - L[i + 1]) / abs(L[1] - L[0]))), 1)
    )
for il in range(len(L) - 1):
    gradVert.append(1.0)

# Mesh stretching
try:
    verticalCoarseningRatio = float(inpt["verticalCoarseningRatio"])
    verticalCoarsening = True
except KeyError:
    verticalCoarsening = False
try:
    radialCoarseningRatio = float(inpt["radialCoarseningRatio"])
    radialCoarsening = True
except KeyError:
    radialCoarsening = False

if verticalCoarsening:
    NVert, gradVert = verticalOutletCoarsening(
        ratio=verticalCoarseningRatio, NVert=NVert, gradVert=gradVert, smooth=True
    )
if radialCoarsening:
    NR, gradR = radialFlowCoarseing(
        ratio=radialCoarseningRatio, NR=NR, gradR=gradR, smooth=True
    )

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
    fw.write("NVert" + str(counter) + " " + str(NVert[counter - 1]) + ";\n")
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
    iwall = amIwall(WallL, WallR, -1, i)
    if iwall == 1:
        fw.write("//")

    i1 = int((i + 1) * 4)
    i2 = int(i * 4)
    fw.write(f"     hex ({i1} {i1+1} {i1+2} {i1+3} {i2} {i2+1} {i2+2} {i2+3})")
    fw.write(f" ($NS1 $NS1 $NVert{i+1}) simpleGrading (1 1 {gradingVert})\n")
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
        iwall = amIwall(WallL, WallR, ir, il)
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
        fw.write(f"     hex ({i1} {i2} {i2+1} {i1+1} {i3} {i4} {i4+1} {i3+1})")
        fw.write(
            f" ($NR{ir+1} $NS1  $NVert{il+1}) simpleGrading ({gradingR} 1 {gradingVert})\n"
        )
        if iwall == 1:
            fw.write("//")
        fw.write(f"     hex ({i1+1} {i2+1} {i2+2} {i1+2} {i3+1} {i4+1} {i4+2} {i3+2})")
        fw.write(
            f" ($NR{ir+1} $NS1  $NVert{il+1}) simpleGrading ({gradingR} 1 {gradingVert})\n"
        )
        if iwall == 1:
            fw.write("//")
        fw.write(f"     hex ({i1+2} {i2+2} {i2+3} {i1+3} {i3+2} {i4+2} {i4+3} {i3+3})")
        fw.write(
            f" ($NR{ir+1} $NS1  $NVert{il+1}) simpleGrading ({gradingR} 1 {gradingVert})\n"
        )
        if iwall == 1:
            fw.write("//")
        fw.write(f"     hex ({i1+3} {i2+3} {i2} {i1} {i3+3} {i4+3} {i4} {i3})")
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
        iwall1 = amIwall(WallL, WallR, ir, il)
        iwall2 = amIwall(WallL, WallR, ir + 1, il)
        iwall3 = amIwall(WallL, WallR, ir, il - 1)
        iwall4 = amIwall(WallL, WallR, ir + 1, il - 1)
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

for i in range(len(BoundaryNames)):
    fw.write("    " + BoundaryNames[i] + "\n")
    fw.write("    " + "{\n")
    fw.write("        " + "type patch;\n")
    fw.write("        " + "faces\n")
    fw.write("        " + "(\n")

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


fw.close()
