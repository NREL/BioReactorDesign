# trace generated using paraview version 5.8.1
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

#### import the simple module from the paraview
import numpy as np
from paraview import simple as pv
import vtk.numpy_interface.dataset_adapter as dsa 
import sys
from sys import argv


solfoam = pv.OpenFOAMReader(FileName = './soln.foam') # just need to provide folder
solfoam.CaseType = 'Reconstructed Case'
solfoam.MeshRegions = ['patch/outlet']
solfoam.CellArrays = ['alpha.gas', 'U.gas', 'thermo:rho.gas','CH4.gas','CO2.gas','H2.gas']
t = np.array(solfoam.TimestepValues)
N=t.size

withsurfnormals = pv.GenerateSurfaceNormals(Input=solfoam)
# Properties modified on calculator1
calculator1 = pv.Calculator(Input=withsurfnormals)
calculator1.AttributeType = 'Point Data'
calculator1.ResultArrayName = 'mfluxg'
calculator1.Function = '"alpha.gas"*("U.gas_X"*"Normals_X"+"U.gas_Y"*"Normals_Y"+"U.gas_Z"*"Normals_Z")*"thermo:rho.gas"'

calculator2 = pv.Calculator(Input=calculator1)
calculator2.AttributeType = 'Point Data'
calculator2.ResultArrayName = 'mfluxco2'
calculator2.Function = '"mfluxg"*"CO2.gas"'

calculator3 = pv.Calculator(Input=calculator2)
calculator3.AttributeType = 'Point Data'
calculator3.ResultArrayName = 'mfluxch4'
calculator3.Function = '"mfluxg"*"CH4.gas"'

calculator4 = pv.Calculator(Input=calculator3)
calculator4.AttributeType = 'Point Data'
calculator4.ResultArrayName = 'mfluxh2'
calculator4.Function = '"mfluxg"*"H2.gas"'

# create a new 'Integrate Variables'
int1 = pv.IntegrateVariables(Input=calculator4)
outfile=open("mflow_outlet.dat","w")
for i in range(N):
    pv.UpdatePipeline(time=t[i], proxy=int1)
    idat    = dsa.WrapDataObject(pv.servermanager.Fetch(int1) )
    area       = idat.CellData['Area'].item()
    mfluxgint   = idat.PointData['mfluxg'].item()*3600.0 #in kg/h
    alphagint   = idat.PointData['alpha.gas'].item()
    mfluxco2int   = idat.PointData['mfluxco2'].item()*3600.0
    mfluxch4int   = idat.PointData['mfluxch4'].item()*3600.0
    mfluxh2int   = idat.PointData['mfluxh2'].item()*3600.0
    print("processing time = %e\t%e\t%e\t%e\t%e" % (t[i],area,alphagint/area,mfluxgint,mfluxco2int))
    outfile.write("%e\t%e\t%e\t%e\t%e\t%e\t%e\n"%(t[i],area,alphagint/area,mfluxgint,mfluxco2int,mfluxch4int,mfluxh2int))

outfile.close()
