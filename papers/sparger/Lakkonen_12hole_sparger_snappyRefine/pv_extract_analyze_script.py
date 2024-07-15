"""
extract data, using Paraview-python modules, to numpy arrays

This script will focus on getting volume-average data for scalar parameters in
the liquid phase

""" 

# H Sitaraman, 2017


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
tt = int(N/2)
print(t)

# threshold filter to get only the "aerated liquid"; specify cell data or point
# data in first element of Scalars by CELLS or POINTS
liquidthreshold = pv.Threshold(Input=ofreader, Scalars=['CELLS', 'alpha.gas'],\
                               ThresholdRange=[0., 0.6])

mwO2 = 2.0 # g/mol
rhol = 1000.0 # kg/m^3 -- could get an average value from data...

# threshold filter to find cells with O2 concentration below a certain value
cO2cut = 0.045 # mol/m^3
yO2cut = cO2cut*mwO2/rhol/1000 # kg/kg
lowO2threshold = pv.Threshold(Input=liquidthreshold,\
                              Scalars=['CELLS', 'H2.liquid'],\
                              ThresholdRange=[0., yO2cut])

# use calculator filter to compute O2 concentration as mass per total unit
# volume
calcfilt = pv.Calculator(Input=liquidthreshold, AttributeType='Cell Data',\
                         ResultArrayName='cO2VT',\
                         Function='H2.liquid*1000.0*alpha.liquid')
calcfilt_fullvolume = pv.Calculator(Input=ofreader, AttributeType='Cell Data',\
                         ResultArrayName='cO2Vfull',\
                         Function='H2.liquid*1000.0*alpha.liquid')

# integrate all variables (in liquid)
integrate_fullvolume = pv.IntegrateVariables(Input=calcfilt_fullvolume)
integrateliq = pv.IntegrateVariables(Input=calcfilt)
# integrate variables in O2-limited region (only need volume -- is there a way
# to limit integration to save compuation? JJS 4/7/16)
integrateO2lim = pv.IntegrateVariables(Input=lowO2threshold)

# get volume-averaged values (in the liquid) as a function of time
# for subsampling the data to reduce post-processing time:
interval = 1 # how often to grab the data; '1' means every timepoint
idx = range(0,N,interval)
tint = t[idx]
nint = tint.size
VT = np.zeros(nint) # total volume; m^3
Vl = np.zeros(nint) # liquid volume; m^3
Vg = np.zeros(nint) # gas volume; m^3
mO2 = np.zeros(nint) # mass O2 in liquid; kg
mO2_fullvol = np.zeros(nint)

# integral volume of O2 threshold
VTO2lim = np.zeros(nint) # total volume; m^3
VlO2lim = np.zeros(nint) # liquid volume; m^3

#update volume of cylinder and liquid volume from geometry
print("processing time = %g" % tint[0])
pv.UpdatePipeline(time=tint[0], proxy=integrateliq)
idat   = dsa.WrapDataObject( pv.servermanager.Fetch(integrateliq) )
VT[0]  = idat.CellData['Volume'].item()
Vl[0]  = idat.CellData['alpha.liquid'].item()
Vg[0]  = idat.CellData['alpha.gas'].item()
mO2[0] = idat.CellData['cO2VT'].item()
pv.UpdatePipeline(time=tint[0], proxy=integrate_fullvolume)
idat   = dsa.WrapDataObject( pv.servermanager.Fetch(integrate_fullvolume) )
mO2_fullvol[0] = idat.CellData['cO2Vfull'].item()

# values
for i in range(1,nint):
    print("processing time = %g" % tint[i])
    pv.UpdatePipeline(time=tint[i], proxy=integrateliq)
    # unfortunately, objects from "Fetch" are not part of the pipeline and must
    # be (re-)called after every update to the time
    idat = dsa.WrapDataObject( pv.servermanager.Fetch(integrateliq) )
    VT[i] = idat.CellData['Volume'].item() 
    Vl[i] = idat.CellData['alpha.liquid'].item()
    Vg[i] = idat.CellData['alpha.gas'].item()
    mO2[i] = idat.CellData['cO2VT'].item()

    pv.UpdatePipeline(time=tint[i], proxy=integrateO2lim)
    idatO2 = dsa.WrapDataObject( pv.servermanager.Fetch(integrateO2lim) )
    # if no cells meet the threshold criteria, idat02 has no data
    if idatO2.CellData.keys():
        VTO2lim[i] = idatO2.CellData['Volume'].item()
        VlO2lim[i] = idatO2.CellData['alpha.liquid'].item()

    pv.UpdatePipeline(time=tint[i], proxy=integrate_fullvolume)
    idat   = dsa.WrapDataObject( pv.servermanager.Fetch(integrate_fullvolume) )
    mO2_fullvol[i] = idat.CellData['cO2Vfull'].item()

# average values
cO2mass = mO2/Vl # kg/m^3
al = Vl/VT # m^3/m^3
ag = Vg/VT # m^3/m^3
# convert O2 concentration to mol/m^3
cO2 = cO2mass/mwO2*1000 # mol/m^3
# fraction of liquid that is O2-limited
O2limfrac = VlO2lim/Vl

outfile=open("volume_avg.dat","w")

for i in range(len(tint)):
	outfile.write("%e\t%e\t%e\t%e\t%e\t%e\n"%(tint[i],cO2[i],mO2[i],mO2_fullvol[i],ag[i],O2limfrac[i]))

outfile.close()
