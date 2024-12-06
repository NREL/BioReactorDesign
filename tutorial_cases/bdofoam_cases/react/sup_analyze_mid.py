"""
extract data, using Paraview-python modules, to numpy arrays

This script is for probing the superificial velocity of the gas (air)

"""

# Jonathan Stickel, Hariswaran Sitaraman; 2016


import os
from sys import argv

import numpy as np
import vtk.numpy_interface.dataset_adapter as dsa
from paraview import simple as pv

holdup = float(argv[1])
# reset stuff

# clear "sources"
for f in pv.GetSources().values():
    pv.Delete(f)

ofreader = pv.OpenFOAMReader(FileName=".")  # just need to provide folder
ofreader.CaseType = "Reconstructed Case"
t = np.array(ofreader.TimestepValues)
N = t.size

# set time to something other than zero to avoid errors about unset fields
pv.UpdatePipeline(time=t[1])

# if needed, evaluate the point locations used in the simulation
ofvtkdata = pv.servermanager.Fetch(ofreader)
ofdata = dsa.WrapDataObject(ofvtkdata)
ofpts = np.array(ofdata.Points.Arrays[0])
ptsmin = ofpts.min(axis=0)  # minimum values of the three axes
ptsmax = ofpts.max(axis=0)  # maximum values of the three axes
print(ptsmin)
print(ptsmax)

# threshold filter to get only the "aerated liquid"; specify cell data or point
# data in first element of Scalars by CELLS or POINTS
liquidthreshold = pv.Threshold(
    Input=ofreader, Scalars=["CELLS", "alpha.gas"], ThresholdRange=[0.0, 0.6]
)

ofvtkdata = pv.servermanager.Fetch(liquidthreshold)
ofdata = dsa.WrapDataObject(ofvtkdata)
ofpts = np.array(ofdata.Points.Arrays[0])
ptsmin_lt = ofpts.min(axis=0)  # minimum values of the three axes
ptsmax_lt = ofpts.max(axis=0)  # maximum values of the three axes
print(ptsmin_lt)
print(ptsmax_lt)

calc1 = pv.Calculator(
    Input=ofreader,
    AttributeType="Cell Data",
    ResultArrayName="vflowrate",
    Function="alpha.gas*U.gas_Y",
)

# if needed, evaluate the point locations used in the simulation
ofvtkdata = pv.servermanager.Fetch(calc1)
ofdata = dsa.WrapDataObject(ofvtkdata)
ofpts = np.array(ofdata.Points.Arrays[0])
ptsmin = ofpts.min(axis=0)  # minimum values of the three axes
ptsmax = ofpts.max(axis=0)  # maximum values of the three axes
print(ptsmin)
print(ptsmax)

# create a new 'Slice'
# yslice = 0.95*ptsmax[1] # near, but not exactly, at the top
yslice = 0.5 * (ptsmax_lt[1] + holdup * ptsmax_lt[1])
print("slice at %g" % yslice)
slice1 = pv.Slice(Input=calc1)
slice1.SliceType.Origin = [0.0, yslice, 0.0]
slice1.SliceType.Normal = [0.0, 1.0, 0.0]

# integrate all variables
int1 = pv.IntegrateVariables(Input=slice1)

# get volume-averaged values (in the liquid) as a function of time
# for subsampling the data to reduce post-processing time:
interval = 1  # how often to grab the data; '1' means every timepoint
idx = range(0, N, interval)
tint = t[idx]
nint = tint.size
sl1_vfrate = np.zeros(nint)
area = np.zeros(nint)
p = np.zeros(nint)

# values
for i in range(0, nint):
    pv.UpdatePipeline(time=tint[i], proxy=int1)
    idat1 = dsa.WrapDataObject(pv.servermanager.Fetch(int1))
    sl1_vfrate[i] = idat1.CellData["vflowrate"].item()
    area[i] = idat1.CellData["Area"].item()
    p[i] = idat1.CellData["p"].item()
    print(
        "processing time = %g" % tint[i],
        sl1_vfrate[i] / area[i],
        p[i] / area[i],
    )

# take average
vs = sl1_vfrate / area
p = p / area  # pressure in the slice

outfile = open("superfvel.dat", "w")

print(
    "average superficial velocity towards later half of time:%e"
    % (np.mean(vs[int(nint / 2) : nint]))
)

for i in range(nint):
    outfile.write("%e\t%e\n" % (tint[i], vs[i]))

outfile.close()

# plt.plot(tint, vs)
# plt.show()
