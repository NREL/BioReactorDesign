from bird.utilities.ofio import *
import os
import numpy as np

def writeGvars(inletA, liqVol):
    filename_tmp = os.path.join("constant", "globalVars_temp")
    with open(filename_tmp,'r+') as f:
        lines = f.readlines()
    filename = os.path.join("constant", "globalVars")
    with open(filename, "w+") as f:
        for line in lines:
            if line.startswith("inletA"):
                f.write(f"inletA\t{inletA:g};\n") 
            elif line.startswith("liqVol"):
                f.write(f"liqVol\t{liqVol:g};\n") 
            else:
                f.write(line)

def readInletArea():
    filename = os.path.join("postProcessing","patchIntegrate(patch=inlet,field=alpha.gas)","0","surfaceFieldValue.dat")
    with open(filename,'r+') as f:
        lines = f.readlines()
    return float(lines[4].split()[-1])

def getLiqVol():
    cellCentres = readMesh(os.path.join(".", f"meshCellCentres_0.obj"))
    volume_field = readOFScal(os.path.join('0','V'), len(cellCentres))
    alpha_field = readOFScal(os.path.join('0','alpha.liquid'), len(cellCentres))
    return np.sum(volume_field*alpha_field)


if __name__ == "__main__":
    A = readInletArea()
    V = getLiqVol()
    writeGvars(A, V)
