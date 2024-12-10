import os
import shutil
import sys

import numpy as np

sys.path.append("util")
import argument_inhomo
from fromMomtoPdf import *


def writeFfield(filename, fieldname, xcent, zcent, schedule, defaultVal, ind):
    fw = open(filename, "w+")
    # Write Header
    fw.write("FoamFile\n")
    fw.write("{\n")
    fw.write("    format      ascii;\n")
    fw.write("    class       volScalarField;\n")
    fw.write(f"    object      {fieldname};\n")
    fw.write("}\n")
    fw.write("\n")
    fw.write("dimensions      [0 0 0 0 0 0 0];\n")
    fw.write("\n")
    fw.write("internalField   uniform 0;\n")
    fw.write("\n")
    fw.write("boundaryField\n")
    fw.write("{\n")
    # All the walls first
    zgBound = ["walls", "outlet"]
    for bound in zgBound:
        fw.write(f"    {bound}\n")
        fw.write("    {" + "\n")
        fw.write("        type            zeroGradient;\n")
        fw.write("    }" + "\n")
    cmBound = ["inlet"]
    for bound in cmBound:
        fw.write(f"    {bound}\n")
        fw.write("    {" + "\n")
        fw.write("        type             codedMixed;\n")
        fw.write(f"        refValue         uniform {defaultVal};\n")
        fw.write("        refGradient      uniform 0;\n")
        fw.write("        valueFraction    uniform 1.0;\n")
        fw.write(f"        redirectType     {fieldname}{bound};\n")
        fw.write(f"        name             {fieldname}{bound};\n")
        fw.write("\n")
        fw.write("        code\n")
        fw.write("        #{\n")
        fw.write("            const fvPatch& boundaryPatch = patch();" + "\n")
        fw.write(
            "            const vectorField& Cf = boundaryPatch.Cf();" + "\n"
        )
        fw.write("            forAll(Cf, faceI)\n")
        fw.write("            {\n")
        fw.write(
            f"                scalar xpos = boundaryPatch.Cf()[faceI][0]-{xcent};\n"
        )
        fw.write(
            f"                scalar zpos = boundaryPatch.Cf()[faceI][2]-{zcent};\n"
        )
        fw.write(
            "                scalar rad = std::sqrt(xpos*xpos + zpos*zpos);\n"
        )
        fw.write(f'                if( rad < {schedule["r_const"]} )\n')
        fw.write("                {\n")
        fw.write(
            f'                    this->refValue()[faceI] = {schedule["f_in"][ind]};\n'
        )
        fw.write("                    this->refGrad()[faceI] = 0.0;\n")
        fw.write("                    this->valueFraction()[faceI] = 1.0;\n")
        fw.write(f'                }} else if (rad < {schedule["r_end"]}){{\n')
        fw.write(
            f'                    this->refValue()[faceI] = {schedule["f_in"][ind]} + (rad - {schedule["r_const"]})/({schedule["r_end"] - schedule["r_const"]}) * ({schedule["f_out"][ind] - schedule["f_in"][ind]});\n'
        )
        fw.write("                    this->refGrad()[faceI] = 0.0;\n")
        fw.write("                    this->valueFraction()[faceI] = 1.0;\n")
        fw.write("                } else {\n")
        fw.write(
            f'                    this->refValue()[faceI] = {schedule["f_out"][ind]};\n'
        )
        fw.write("                    this->refGrad()[faceI] = 0.0;\n")
        fw.write("                    this->valueFraction()[faceI] = 1.0;\n")
        fw.write("                }\n")
        fw.write("            }\n")
        fw.write("        #};\n")
        fw.write("        codeInclude\n")
        fw.write("        #{\n")
        fw.write("        #};\n")
        fw.write("\n")
        fw.write("        codeOptions" + "\n")
        fw.write("        #{" + "\n")
        fw.write("        #};" + "\n")
        fw.write("    }" + "\n")

        fw.write("}\n")
        fw.close()


def binInfo(phasePropFile):
    f = open(phasePropFile, "r")
    nFields = 0
    diam = []
    value = []
    fname = []
    lines = f.readlines()
    for iline, line in enumerate(lines):
        if "sizeGroups" in line.strip():
            startLine = iline
            break
    for line in lines[iline + 1 :]:
        if line.strip().startswith("f"):
            nFields += 1
            fname.append(line.strip().split()[0])
            indSem = line.strip().find(";")
            indSem2 = line.strip()[indSem + 1 :].find(";")
            indD = line.strip().find("dSph", 0, indSem)
            try:
                diamStr = line.strip()[indD:indSem].split()[1]
                diam.append(float(diamStr))
            except ValueError:
                print(f"ERROR: Could not convert {diamStr} to float")
            try:
                valStr = line.strip()[indSem + 1 :][:indSem2].split()[1]
                value.append(float(valStr))
            except ValueError:
                print(f"ERROR: Could not convert {valStr} to float")
        elif line.strip().startswith("(") or line.strip().startswith(")"):
            pass
        elif line.strip().startswith("}"):
            break
    f.close()
    bin_size = np.diff(np.array(diam))
    ave_bin_size = np.mean(bin_size)
    std_bin_size = np.std(bin_size)
    if std_bin_size > (ave_bin_size * 1e-6):
        print(f"ERROR: Do not use with non uniform bin spacing")
    return nFields, diam, ave_bin_size, fname, value


if __name__ == "__main__":
    args = argument_inhomo.initArgs()
    print("ASSUMPTIONS:")
    print("\t-constant bin size")
    print("\t-y axis is the gravity axis")
    print("\t-diameters are ordered\n")

    schedule = {
        "r_const": args.r_const,
        "r_end": args.r_end,
        "pore_in": args.pore_in,
        "pore_out": args.pore_out,
    }

    phasePropFile = "constant/phaseProperties"
    nf, diam, bin_size, fname, value = binInfo(phasePropFile)

    meanTar, stdTar = poreDiamCorr(
        dp=schedule["pore_in"], ds=args.diam_sparger, Ugs=args.superf_vel
    )
    f_in = get_f_vals(meanTar, stdTar, np.array(diam), verb=args.verbose)
    meanTar, stdTar = poreDiamCorr(
        dp=schedule["pore_out"], ds=args.diam_sparger, Ugs=args.superf_vel
    )
    f_out = get_f_vals(meanTar, stdTar, np.array(diam), verb=args.verbose)

    schedule["f_in"] = f_in
    schedule["f_out"] = f_out

    try:
        shutil.rmtree("IC_inhomo")
    except FileNotFoundError as err:
        print(err)
        pass
    shutil.copytree("IC", "IC_inhomo")

    for val, name in zip(value, fname):
        fieldname = f"{name}.gas"
        filename = os.path.join("IC_inhomo", "0", f"{name}.gas")
        ind = fname.index(name)
        writeFfield(
            filename, fieldname, args.xcent, args.zcent, schedule, val, ind
        )
