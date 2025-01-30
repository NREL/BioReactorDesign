"""
extract data, using Paraview-python modules, to numpy arrays

This script will focus on getting volume-average data for scalar parameters in
the liquid phase

""" 
import numpy as np
from paraview import simple as pv
import vtk.numpy_interface.dataset_adapter as dsa 
import sys
from scipy.optimize import curve_fit

def func(t,cstar,kla):
    global t0,c0
    return( (cstar-c0)*(1-np.exp(-kla*(t-t0)))+c0 )

data=np.loadtxt("volume_avg.dat",dtype=float)

params=5,0.01
t0=data[0,0]
c0=data[0,1]
t=data[:,0]
cH2=data[:,1]
fitparamsH2,cov=curve_fit(func,t,cH2, params)
cH2fit=func(t,*fitparamsH2)

#c0=data[0,2]
#cCO2=data[:,2]
#fitparamsCO2,cov=curve_fit(func,t,cCO2, params)
#cCO2fit=func(t,*fitparamsCO2)

np.savetxt("fitting.dat",np.transpose(np.vstack((t,cH2,cH2fit))),delimiter="  ")

np.savetxt("cstar_kla.dat",np.vstack((fitparamsH2)),delimiter="   ")
print("cstar,kla O2:",fitparamsH2[0],fitparamsH2[1])
#print("cstar,kla CO2:",fitparamsCO2[0],fitparamsCO2[1])
