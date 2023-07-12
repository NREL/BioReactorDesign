import numpy as np
import argparse
import sys
sys.path.append('util')
import os
from ofio import *

parser = argparse.ArgumentParser(description="Case folder")
parser.add_argument(
    "-f",
    "--caseFolder",
    type=str,
    metavar="",
    required=True,
    help="caseFolder to analyze",
    default=None,
)

parser.add_argument(
    "-avg",
    "--windowAve",
    type=int,
    metavar="",
    required=False,
    help="Window Average",
    default=1,
)


args = parser.parse_args()


case_path = args.caseFolder
time_float_sorted, time_str_sorted = getCaseTimes(case_path)
cellCentres = readMesh(os.path.join(case_path,'meshCellCentres_0.obj'))
nCells = len(cellCentres)

window_ave = min(args.windowAve,len(time_str_sorted))

nbins = 32

for i_ave in range(window_ave):
    time_folder = time_str_sorted[-i_ave-1]
    alpha_gas_file = os.path.join(case_path,time_folder,'alpha.gas')
    co2_gas_file = os.path.join(case_path,time_folder,'CO2.gas')
    d_gas_file = os.path.join(case_path,time_folder,'d.gas')
    if os.path.isfile(d_gas_file):
        has_d = True
    else:
        has_d = False

    alpha_gas = readOFScal(alpha_gas_file,nCells)
    co2_gas = readOFScal(co2_gas_file,nCells) 
    if has_d:
        d_gas = readOFScal(d_gas_file,nCells)
        y_axis, d_cond_tmp = conditionalAverage(cellCentres[:,1], d_gas, nbin=nbins)
    else:
        d_cond_tmp = np.ones(nbins)*2.86e-3
    y_axis, alpha_cond_tmp = conditionalAverage(cellCentres[:,1], alpha_gas, nbin=nbins)
    y_axis, co2_cond_tmp = conditionalAverage(cellCentres[:,1], co2_gas, nbin=nbins)    
    
    if i_ave == 0:
        alpha_cond = alpha_cond_tmp
        co2_cond = co2_cond_tmp
        d_cond = d_cond_tmp
    else:
        alpha_cond += alpha_cond_tmp
        co2_cond += co2_cond_tmp
        d_cond += d_cond_tmp
alpha_cond /= window_ave
co2_cond /= window_ave
d_cond /= window_ave

np.savez(os.path.join(case_path, 'observations.npz'), 
         z=y_axis,
         gh=alpha_cond,
         co2=co2_cond,
         d=d_cond)
