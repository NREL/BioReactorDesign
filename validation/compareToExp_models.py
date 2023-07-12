import numpy as np
import argparse
import sys
sys.path.append('util')
import os
from ofio import *
from plotsUtil import *
from folderManagement import *

parser = argparse.ArgumentParser(description="Case folder")
parser.add_argument(
    "-rd",
    "--refData",
    type=str,
    metavar="",
    required=True,
    help="reference data",
    default=None,
)
parser.add_argument(
    "-f17",
    "--caseFolder17",
    type=str,
    metavar="",
    required=False,
    help="caseFolder17 to analyze",
    nargs='+',
    default=None,
)
parser.add_argument(
    "-f19",
    "--caseFolder19",
    type=str,
    metavar="",
    required=False,
    help="caseFolder19 to analyze",
    nargs='+',
    default=None,
)
parser.add_argument(
    "-ff",
    "--figureFolder",
    type=str,
    metavar="",
    required=False,
    help="figureFolder",
    default=None,
)
parser.add_argument(
    "-n17",
    "--names17",
    type=str,
    metavar="",
    required=False,
    help="names of cases 17",
    nargs='+',
    default=None,
)
parser.add_argument(
    "-n19",
    "--names19",
    type=str,
    metavar="",
    required=False,
    help="names of cases 19",
    nargs='+',
    default=None,
)

args = parser.parse_args()

figureFolder = 'Figures'
figureFolder = os.path.join(figureFolder, args.figureFolder)
makeRecursiveFolder(figureFolder)

if args.caseFolder17 is not None:
    if isinstance(args.caseFolder17, str):
        caseFolder17 = args.caseFolder17.split()
    else:
        caseFolder17 = args.caseFolder17
    n17 = len(caseFolder17)
else:
    caseFolder17 = []
    n17 = 0

if args.caseFolder19 is not None:
    if isinstance(args.caseFolder19, str):
        caseFolder19 = args.caseFolder19.split()
    else:
        caseFolder19 = args.caseFolder19
    n19 = len(caseFolder19)
else:
    caseFolder19 = []
    n19 = 0

if args.names17 is None or n17<=1:
    names17=None
elif args.names17 is None:
    names17=['case'+str(i) for i in range(len(n17))]
else:
    if isinstance(args.names17, str):
        names17=args.names17.split()
    else:
        names17=args.names17
    
if args.names19 is None or n19<=1:
    names19=None
elif args.names19 is None:
    names19=['case'+str(i) for i in range(len(n19))]
else:
    if isinstance(args.names19, str):
        names19=args.names19.split()
    else:
        names19=args.names19

symbList = ['-','-d','-^','-.', '-s', '-o', '-+']
if n17>len(symbList) or n19>len(symbList): 
    print(f"ERROR: too many cases (case17: {n17}, case19: {n19}), reduce number of case to {len(symbList)} or add symbols")
    sys.exit()    

conv17 = []
conv19 = []
sim17 = []
sim19 = []

def convertToMol(yCO2):
    MN2 = 0.028
    MCO2 = 0.044
    return MN2 * yCO2 / (MCO2 + MN2 * yCO2 - MCO2 * yCO2)

for folder17 in caseFolder17:
    conv17.append(np.load(os.path.join(folder17,'convergence_gh.npz')))
    sim17.append(np.load(os.path.join(folder17,'observations.npz')))

for folder19 in caseFolder19:
    conv19.append(np.load(os.path.join(folder19,'convergence_gh.npz')))
    sim19.append(np.load(os.path.join(folder19,'observations.npz')))

refData = np.load(args.refData)


def sequencePlot(seq,xlab,ylab, mode, xfun=None, yfun=None):
    
    if str(mode)=='17':
        n=n17
        labelStart = 'Sim17'
        color='r'
        names = names17
    elif str(mode)=='19':
        n=n19
        labelStart = 'Sim19'
        color='b'
        names = names19

    if n>=1:
        for ic, c in enumerate(seq):
            label=''
            if ic==0:
                label += labelStart + ' ' + names[ic]
            else:
                label += '' +  names[ic]
            if xfun is None:
               xval = c[xlab]
            else:
               xval = xfun(c[xlab])
            if yfun is None:
               yval = c[ylab]
            else:
               yval = yfun(c[ylab])
            plt.plot(xval,yval, symbList[ic], markersize=10, markevery=10, linewidth=3, color=color, label=label)
        


fig = plt.figure()
sequencePlot(conv17,'time', 'gh', 17)
sequencePlot(conv19,'time', 'gh', 19)
plotLegend()
prettyLabels('t [s]', 'Gas Holdup', 14)
plt.savefig(os.path.join(figureFolder,'conv.png'))
plt.close()

 
fig = plt.figure()
if n17>0:
    plt.plot(refData['gh_exp17'][:,0], refData['gh_exp17'][:,1], 's', color='k', label='exp17')
    plt.plot(refData['gh_exp17_ngu'][:,0], refData['gh_exp17_ngu'][:,1], color='k', label='Ngu 17')
    plt.plot(refData['gh_exp17_hassaniga'][:,0], refData['gh_exp17_hassaniga'][:,1], '--', color='k', label='Hissanaga 17')
if n19>0:
    plt.plot(refData['gh_exp19'][:,0], refData['gh_exp19'][:,1], 'o', color='k', label='exp19')
    plt.plot(refData['gh_exp19_ngu'][:,0], refData['gh_exp19_ngu'][:,1], color='k', label='Ngu 19')
    plt.plot(refData['gh_exp19_hassaniga'][:,0], refData['gh_exp19_hassaniga'][:,1], '--', color='k', label='Hissanaga 19')
sequencePlot(sim19,'gh', 'z', 19)
sequencePlot(sim17,'gh', 'z', 17)
plotLegend()
prettyLabels('Gas Holdup', 'z [m]', 14)
plt.savefig(os.path.join(figureFolder,'gh.png'))
plt.close()

fig = plt.figure()
if n17>0:
    plt.plot(refData['xco2_exp17'][:,0], refData['xco2_exp17'][:,1], 's', color='k', label='exp17')
    plt.plot(refData['xco2_exp17_ngu'][:,0], refData['xco2_exp17_ngu'][:,1], color='k', label='Ngu 17')
    plt.plot(refData['xco2_exp17_hassaniga'][:,0], refData['xco2_exp17_hassaniga'][:,1], '--', color='k', label='Hissanaga 17')
if n19>0:
    plt.plot(refData['xco2_exp19'][:,0], refData['xco2_exp19'][:,1], 'o', color='k', label='exp19')
    plt.plot(refData['xco2_exp19_ngu'][:,0], refData['xco2_exp19_ngu'][:,1], color='k', label='Ngu 19')
    plt.plot(refData['xco2_exp19_hassaniga'][:,0], refData['xco2_exp19_hassaniga'][:,1], '--', color='k', label='Hissanaga 19')
sequencePlot(sim19,'co2', 'z', 19, xfun=convertToMol)
sequencePlot(sim17,'co2', 'z', 17, xfun=convertToMol)
plotLegend()
prettyLabels(r'$X_{CO2}$', 'z [m]', 14)
plt.savefig(os.path.join(figureFolder,'co2.png'))
plt.close()

fig = plt.figure()
sequencePlot(sim19,'d', 'z', 19)
sequencePlot(sim17,'d', 'z', 17)
plotLegend()
prettyLabels(r'diam [m]', 'z [m]', 14)
plt.savefig(os.path.join(figureFolder,'d.png'))
plt.close()
