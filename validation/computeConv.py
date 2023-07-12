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

args = parser.parse_args()


case_path = args.caseFolder
time_float_sorted, time_str_sorted = getCaseTimes(case_path)
cellCentres = readMesh(os.path.join(case_path,'meshCellCentres_0.obj'))
nCells = len(cellCentres)

gh_history = np.zeros(len(time_str_sorted)-1)
for i in range(len(time_str_sorted)-1):
    print("t = ",  time_str_sorted[i+1])
    gh_history[i] = compute_gas_holdup(case_path, time_str_sorted[i+1], nCells)

np.savez(os.path.join(case_path, 'convergence_gh.npz'), 
        time=time_float_sorted[1:],
        gh=gh_history)
