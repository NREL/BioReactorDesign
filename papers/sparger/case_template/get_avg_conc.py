"""
extract data, using Paraview-python modules, to numpy arrays

This script will focus on getting volume-average data for scalar parameters in
the liquid phase

""" 
import numpy as np
from paraview import simple as pv
import vtk.numpy_interface.dataset_adapter as dsa 
import sys


ofreader = pv.OpenFOAMReader(FileName = './soln.foam') # just need to provide folder
ofreader.CaseType = 'Reconstructed Case'
ofreader.MeshRegions = ['internalMesh']
ofreader.SkipZeroTime = 0  #dont know why this is not working

t = np.array(ofreader.TimestepValues)
N = t.size
print(t)

# threshold filter to get only the "aerated liquid"; specify cell data or point
# data in first element of Scalars by CELLS or POINTS
liquidthreshold = pv.Threshold(Input=ofreader, Scalars=['CELLS', 'alpha.gas'],\
                               ThresholdRange=[0., 0.6])

calc1 = pv.Calculator(Input=liquidthreshold, AttributeType='Cell Data',\
                         ResultArrayName='cH2',\
                         Function='H2.liquid*1000.0*alpha.liquid/0.002')
calc2 = pv.Calculator(Input=calc1, AttributeType='Cell Data',\
                         ResultArrayName='cCO',\
                         Function='CO.liquid*1000.0*alpha.liquid/0.028')
calc3 = pv.Calculator(Input=calc2, AttributeType='Cell Data',\
                         ResultArrayName='cCO2',\
                         Function='CO2.liquid*1000.0*alpha.liquid/0.044')

# integrate all variables (in liquid)
integrateliq = pv.IntegrateVariables(Input=calc3)

Vl = np.zeros(N) # liquid volume; m^3
molH2 = np.zeros(N) # H2 in liquid; moles
molCO = np.zeros(N) # H2 in liquid; moles
molCO2 = np.zeros(N) # H2 in liquid; moles
bubdia=np.zeros(N)
Vg = np.zeros(N)
Vt = np.zeros(N)


for i in range(N):
    print("processing time = %g" % t[i])
    pv.UpdatePipeline(time=t[i], proxy=integrateliq)
    idat = dsa.WrapDataObject( pv.servermanager.Fetch(integrateliq) )
    Vt[i] = idat.CellData['Volume'].item() 
    Vl[i] = idat.CellData['alpha.liquid'].item()
    Vg[i] = idat.CellData['alpha.gas'].item()
    molH2[i] = idat.CellData['cH2'].item()
    molCO[i] = idat.CellData['cCO'].item()
    molCO2[i] = idat.CellData['cCO2'].item()
    bubdia[i] = idat.CellData['d.gas'].item()

ag = Vg/Vt # m^3/m^3
cH2 = molH2/Vl
cCO = molCO/Vl
cCO2 = molCO2/Vl
bubdia=bubdia/Vt

np.savetxt("volume_avg.dat",np.transpose(np.vstack((t,cH2,cCO,cCO2,ag,bubdia))),delimiter="  ")
