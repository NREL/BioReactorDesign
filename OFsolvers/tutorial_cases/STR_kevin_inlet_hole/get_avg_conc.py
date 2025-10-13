"""
extract data, using Paraview-python modules, to numpy arrays

This script will focus on getting volume-average data for scalar parameters in
the liquid phase

""" 
import numpy as np
from paraview import simple as pv
import vtk.numpy_interface.dataset_adapter as dsa 
import sys


ofreader = pv.OpenFOAMReader(FileName = '.') # just need to provide folder
ofreader.CaseType = 'Reconstructed Case'
ofreader.MeshRegions = ['internalMesh']
ofreader.SkipZeroTime = 0  #dont know why this is not working

t = np.array(ofreader.TimestepValues)
N = t.size
print(t)

# threshold filter to get only the "aerated liquid"; specify cell data or point
# data in first element of Scalars by CELLS or POINTS
liquidthreshold = pv.Threshold(Input=ofreader, Scalars=['CELLS', 'alpha.gas'],\
#                               ThresholdRange=[0., 0.6])
                               LowerThreshold=0,UpperThreshold=0.6,ThresholdMethod='Between')

# integrate all variables (in liquid)
integrateliq = pv.IntegrateVariables(Input=liquidthreshold)

Vl = np.zeros(N) # liquid volume; m^3
Vg = np.zeros(N)
Vt = np.zeros(N)


for i in range(N):
    print("processing time = %g" % t[i])
    pv.UpdatePipeline(time=t[i], proxy=integrateliq)
    idat = dsa.WrapDataObject( pv.servermanager.Fetch(integrateliq) )
    Vt[i] = idat.CellData['Volume'].item() 
    Vl[i] = idat.CellData['alpha.liquid'].item()
    Vg[i] = idat.CellData['alpha.gas'].item()

ag = Vg/Vt # m^3/m^3

np.savetxt("volume_avg.dat",np.transpose(np.vstack((t,ag))),delimiter="  ")
