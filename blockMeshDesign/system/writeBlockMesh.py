import numpy as np
import os
import sys

def fun(G,N1):
    result = (1.0-G)/(G*(1-np.power(G,1.0/N1)))
    return result
#def fun(G,N1):
#    result = (1.0-G**(N1/(N1-1)))/((G**(1.0+1.0/(N1-1)))*(1-np.power(G,1.0/(N1-1))))
#    return result

def bissection(val,fun,N1):
    Gmin = 0.01
    Gmax = 100
    resultmin = fun(Gmin,N1)-val
    resultmax = fun(Gmax,N1)-val
    if resultmin*resultmax>0:
        print('Error,the initial bounds of grading do not encompass the solution')
        #stop

    for i in range(100):
        Gmid = 0.5*(Gmax+Gmin)
        resultmid = fun(Gmid,N1)-val
        if resultmid*resultmax<0:
            Gmin = Gmid
            resultmin = resultmid
        else:
            Gmax = Gmid
            resultmax = resultmid

    return Gmid

def amIwall(WallL,WallR,ir,il):
    # returns 1 if block is wall
    # returns 0 otherwise
    iwall = 0
    for iw in range(len(WallL)):
       if WallL[iw] == (il+1) and WallR[iw] == ir+1:
          iwall = 1
    return iwall

def mergeSort(list,reverse):
    list.sort(reverse=reverse)
    listtmp = []
    for val in list:
        if len(listtmp)==0 or not val==listtmp[-1]:
            listtmp.append(val)
    return listtmp

# ~~~ Initialize input
input_file = sys.argv[1]

# ~~~~ Parse input
inpt = {}
f = open( input_file )
data = f.readlines()
for line in data:
    if ':' in line:
        key, value = line.split(":")
        inpt[key.strip()] = value.strip()
f.close()

# ~~~~ Define dimensions based on input
RIBoat      =  float(inpt[ 'RIBoat'  ])
ROBoat      =  float(inpt[ 'ROBoat'  ])
RHeater     =  float(inpt[ 'RHeater'  ])
RPlatter    =  float(inpt[ 'RPlatter'  ])
RIFurnace   =  float(inpt[ 'RIFurnace'  ])
ROFurnace   =  float(inpt[ 'ROFurnace'  ])
RICurtain1  =  float(inpt[ 'RICurtain1'  ])
ROCurtain1  =  float(inpt[ 'ROCurtain1'  ])
RICurtain2  =  float(inpt[ 'RICurtain2'  ])
ROCurtain2  =  float(inpt[ 'ROCurtain2'  ])
RIDump      =  float(inpt[ 'RIDump'  ])
RODump      =  float(inpt[ 'RODump'  ])
RICurtainDump =  float(inpt[ 'RICurtainDump'  ])
ROCurtainDump =  float(inpt[ 'ROCurtainDump'  ])
Rout        =  float(inpt[ 'Rout'  ])

Ltop            =  float(inpt[ 'Ltop'  ])
LboatTop        =  float(inpt[ 'LboatTop'  ])
LboatBottom     =  float(inpt[ 'LboatBottom'  ])
LOutletTop      =  float(inpt[ 'LOutletTop'  ])
LPlatter        =  float(inpt[ 'LPlatter'  ])
LPlatterBottom  =  float(inpt[ 'LPlatterBottom'  ])
LHeaterBottom   =  float(inpt[ 'LHeaterBottom'  ])
LFurnace        =  float(inpt[ 'LFurnace'  ])
LCurtain1       =  float(inpt[ 'LCurtain1'  ])
LCurtain2       =  float(inpt[ 'LCurtain2'  ])
LCurtainDump    =  float(inpt[ 'LCurtainDump'  ])
LDump           =  float(inpt[ 'LDump'  ])
LSupport        =  float(inpt[ 'LSupport'  ])
    
verticalDir        =  inpt[ 'verticalDir'  ]

outfile = 'blockMeshDict'



# Dimensions
R = [RIBoat, ROBoat, RHeater, RPlatter, RIFurnace, ROFurnace, RICurtain1, ROCurtain1, RICurtain2, ROCurtain2, RIDump, RODump, RICurtainDump, ROCurtainDump, Rout]
L = [Ltop, LboatTop, LboatBottom, LOutletTop, LPlatter, LPlatterBottom, LHeaterBottom, LFurnace, LCurtain1, LCurtain2, LCurtainDump, LDump, LSupport]

# Merge and sort R and L
R = mergeSort(R,False)
L = mergeSort(L,True)

# Define blocks that will be walls
WallR=[]
WallL=[] 
#Boat
WallR.append(2)
WallL.append(2)
WallR.append(3)
WallL.append(2)
# FurnaceSides
WallR.append(5)
WallL.append(1)
WallR.append(5)
WallL.append(2)
WallR.append(5)
WallL.append(3)
WallR.append(5)
WallL.append(4)
WallR.append(5)
WallL.append(5)
WallR.append(5)
WallL.append(6)
WallR.append(5)
WallL.append(7)
## Curtain1
WallR.append(7)
WallL.append(1)
WallR.append(7)
WallL.append(2)
WallR.append(7)
WallL.append(3)
WallR.append(7)
WallL.append(4)
WallR.append(7)
WallL.append(5)
WallR.append(7)
WallL.append(6)
WallR.append(7)
WallL.append(7)
## Curtain2
WallR.append(9)
WallL.append(1)
WallR.append(9)
WallL.append(2)
WallR.append(9)
WallL.append(3)
WallR.append(9)
WallL.append(4)
WallR.append(9)
WallL.append(5)
WallR.append(9)
WallL.append(6)
WallR.append(9)
WallL.append(7)
## Outside
WallR.append(10)
WallL.append(1)
WallR.append(10)
WallL.append(2)
WallR.append(10)
WallL.append(3)
## Plate
WallR.append(0)
WallL.append(5)
WallR.append(1)
WallL.append(5)
WallR.append(2)
WallL.append(5)
## Backside Heater
WallR.append(0)
WallL.append(6)
WallR.append(1)
WallL.append(6)
WallR.append(2)
WallL.append(6)
# Plate Support
WallR.append(0)
WallL.append(7)
WallR.append(1)
WallL.append(7)
WallR.append(0)
WallL.append(8)
WallR.append(1)
WallL.append(8)
WallR.append(0)
WallL.append(9)
WallR.append(1)
WallL.append(9)
# Dump
WallR.append(5)
WallL.append(9)
# Curtain Dump
WallR.append(7)
WallL.append(9)

# Define boundaries
BoundaryNames=[]
BoundaryType=[]
BoundaryRmin=[]
BoundaryRmax=[]
BoundaryLmin=[]
BoundaryLmax=[]
BoundaryNames.append('Boat')
BoundaryType.append(['lateral','lateral', 'top', 'top', 'bottom', 'bottom'])
BoundaryRmin.append([1,3,2,3,2,3])
BoundaryRmax.append([2,4,2,3,2,3])
BoundaryLmin.append([2,2,1,1,2,2])
BoundaryLmax.append([2,2,2,2,3,3])
BoundaryNames.append('FurnaceSides')
BoundaryType.append(['bottom'])
BoundaryRmin.append([5])
BoundaryRmax.append([5])
BoundaryLmin.append([7])
BoundaryLmax.append([8])
BoundaryNames.append('FurnaceSides_Chamber')
BoundaryType.append(['lateral','lateral','lateral', 'lateral', 'lateral', 'lateral', 'lateral'])
BoundaryRmin.append([4,4,4,4,4,4,4])
BoundaryRmax.append([5,5,5,5,5,5,5])
BoundaryLmin.append([1,2,3,4,5,6,7])
BoundaryLmax.append([1,2,3,4,5,6,7])
BoundaryNames.append('FurnaceSides_Curtain1')
BoundaryType.append(['lateral','lateral','lateral','lateral','lateral', 'lateral', 'lateral'])
BoundaryRmin.append([5,5,5,5,5,5,5])
BoundaryRmax.append([6,6,6,6,6,6,6])
BoundaryLmin.append([1,2,3,4,5,6,7])
BoundaryLmax.append([1,2,3,4,5,6,7])
BoundaryNames.append('FurnaceTop')
BoundaryType.append(['top','top','top','top','top'])
BoundaryRmin.append([0,1,2,3,4])
BoundaryRmax.append([0,1,2,3,4])
BoundaryLmin.append([0,0,0,0,0])
BoundaryLmax.append([1,1,1,1,1])
BoundaryNames.append('Curtain1')
BoundaryType.append(['bottom'])
BoundaryRmin.append([7])
BoundaryRmax.append([7])
BoundaryLmin.append([7])
BoundaryLmax.append([8])
BoundaryNames.append('Curtain1_FurnaceSides')
BoundaryType.append(['lateral','lateral', 'lateral', 'lateral', 'lateral','lateral', 'lateral'])
BoundaryRmin.append([6,6,6,6,6,6,6])
BoundaryRmax.append([7,7,7,7,7,7,7])
BoundaryLmin.append([1,2,3,4,5,6,7])
BoundaryLmax.append([1,2,3,4,5,6,7])
BoundaryNames.append('Curtain1_Curtain2')
BoundaryType.append(['lateral', 'lateral', 'lateral', 'lateral', 'lateral','lateral', 'lateral'])
BoundaryRmin.append([7,7,7,7,7,7,7])
BoundaryRmax.append([8,8,8,8,8,8,8])
BoundaryLmin.append([1,2,3,4,5,6,7])
BoundaryLmax.append([1,2,3,4,5,6,7])
BoundaryNames.append('Curtain2')
BoundaryType.append(['lateral', 'lateral','lateral', 'lateral', 'bottom'])
BoundaryRmin.append([ 9, 9, 9, 9, 9])
BoundaryRmax.append([10,10,10,10, 9])
BoundaryLmin.append([4 , 5, 6, 7, 7])
BoundaryLmax.append([4 , 5, 6, 7, 8])
BoundaryNames.append('Curtain2_Curtain1')
BoundaryType.append(['lateral','lateral', 'lateral', 'lateral', 'lateral', 'lateral', 'lateral'])
BoundaryRmin.append([ 8, 8, 8, 8, 8, 8, 8])
BoundaryRmax.append([ 9, 9, 9, 9, 9, 9, 9])
BoundaryLmin.append([ 1, 2, 3, 4, 5, 6, 7])
BoundaryLmax.append([ 1, 2, 3, 4, 5, 6, 7])
BoundaryNames.append('Dump')
BoundaryType.append(['lateral','lateral', 'top'])
BoundaryRmin.append([ 4, 5, 5])
BoundaryRmax.append([ 5, 6, 5])
BoundaryLmin.append([ 9, 9, 8])
BoundaryLmax.append([ 9, 9, 9])
BoundaryNames.append('CurtainDump')
BoundaryType.append(['lateral', 'lateral', 'top'])
BoundaryRmin.append([ 6, 7, 7])
BoundaryRmax.append([ 7, 8, 7])
BoundaryLmin.append([ 9, 9, 8])
BoundaryLmax.append([ 9, 9, 9])
BoundaryNames.append('PlatterTop')
BoundaryType.append(['top', 'top', 'top'])
BoundaryRmin.append([0,1,2])
BoundaryRmax.append([0,1,2])
BoundaryLmin.append([4,4,4])
BoundaryLmax.append([5,5,5])
BoundaryNames.append('PlatterOther')
BoundaryType.append(['lateral'])
BoundaryRmin.append([2])
BoundaryRmax.append([3])
BoundaryLmin.append([5])
BoundaryLmax.append([5])
BoundaryNames.append('Heater')
BoundaryType.append(['lateral','bottom'])
BoundaryRmin.append([2, 2])
BoundaryRmax.append([3, 2])
BoundaryLmin.append([6, 6])
BoundaryLmax.append([6, 7])
BoundaryNames.append('Support')
BoundaryType.append(['lateral', 'lateral', 'lateral'])
BoundaryRmin.append([1, 1, 1])
BoundaryRmax.append([2, 2, 2])
BoundaryLmin.append([7, 8, 9])
BoundaryLmax.append([7, 8, 9])
BoundaryNames.append('OutsideTop')
BoundaryType.append(['bottom'])
BoundaryRmin.append([10])
BoundaryRmax.append([10])
BoundaryLmin.append([ 3])
BoundaryLmax.append([ 4])
BoundaryNames.append('InflowCurtain1')
BoundaryType.append(['top'])
BoundaryRmin.append([6])
BoundaryRmax.append([6])
BoundaryLmin.append([0])
BoundaryLmax.append([1])
BoundaryNames.append('InflowCurtain2')
BoundaryType.append(['top'])
BoundaryRmin.append([8])
BoundaryRmax.append([8])
BoundaryLmin.append([0])
BoundaryLmax.append([1])
BoundaryNames.append('Outflow')
BoundaryType.append(['bottom','bottom','bottom'])
BoundaryRmin.append([ 2, 3, 4])
BoundaryRmax.append([ 2, 3, 4])
BoundaryLmin.append([ 9, 9, 9])
BoundaryLmax.append([10,10,10])
BoundaryNames.append('Outflow_outside')
BoundaryType.append(['bottom','bottom','bottom'])
BoundaryRmin.append([ 8, 9,10])
BoundaryRmax.append([ 8, 9,10])
BoundaryLmin.append([ 9, 9, 9])
BoundaryLmax.append([10,10,10])
BoundaryNames.append('OutflowSide')
BoundaryType.append(['lateral','lateral','lateral','lateral','lateral','lateral'])
BoundaryRmin.append([10,10,10,10,10,10])
BoundaryRmax.append([11,11,11,11,11,11])
BoundaryLmin.append([ 4, 5, 6, 7, 8, 9])
BoundaryLmax.append([ 4, 5, 6, 7, 8, 9])
BoundaryNames.append('InflowCurtainDump')
BoundaryType.append(['bottom'])
BoundaryRmin.append([6])
BoundaryRmax.append([6])
BoundaryLmin.append([9])
BoundaryLmax.append([10])

N1 = len(R)
N2 = len(L)-1
CW=[]
mCW=[]
C1=[]
mC1=[]
C2=[]
mC2=[]
for rval in R:
    CW.append(rval*0.5)
    mCW.append(-rval*0.5)
    C1.append(rval*np.cos(np.pi/4))
    mC1.append(-rval*np.cos(np.pi/4))
    C2.append(rval*np.sin(np.pi/4))
    mC2.append(-rval*np.sin(np.pi/4))

Refinement = float(inpt[ 'Refinement'  ])
NS = [] 
NR = [] 
NVert = []
gradR_l = [] 
gradR_r = [] 
gradR = [] 
gradVert = []


NS.append(max(int(round(16*Refinement/0.3)),12))
#for i in range(len(R)-1):
#    NS.append(int(np.floor(NS[0]*abs(R[i+1]-R[i])/abs(R[0]-R[0]/2))))
NR.append(int(NS[0]/2))
for i in range(len(R)-1):
    NR.append(max(int(round(NR[0]*abs(R[i+1]-R[i])/abs(R[0]-R[0]/2))),1))
#At least three points through dump Wall
wallR = 8
NR[wallR]=int(round(3*Refinement/0.3))
# Adjust resolutions of other elements
# outlet is coarser
outletR = 9
NR[outletR] = min(max(int(np.floor(NR[outletR-1]*abs(R[outletR]-R[outletR-1])/(1.5*abs(R[outletR-1]-R[outletR-2])))),1),25)

# These near wall element are not coarser
#NR[ 11] = max(int(int(round(NR[wallR]*abs(R[11 ]-R[10 ])/abs(R[wallR]-R[wallR-1])))),1)
#NR[ 10] = max(int(int(round(NR[wallR]*abs(R[10 ]-R[9 ])/abs(R[wallR]-R[wallR-1])))),8)
#NR[ 9] = max(int(int(round(NR[wallR]*abs(R[9 ]-R[8 ])/abs(R[wallR]-R[wallR-1])))),16)
NR[ 8] = max(int(int(round(NR[wallR]*abs(R[8 ]-R[7 ])/abs(R[wallR]-R[wallR-1])))),8)
NR[ 7] = max(int(int(round(NR[wallR]*abs(R[7 ]-R[6 ])/abs(R[wallR]-R[wallR-1])))),16)
NR[ 6] = max(int(int(round(NR[wallR]*abs(R[6 ]-R[5 ])/abs(R[wallR]-R[wallR-1])))),8)
NR[ 5] = max(int(int(round(NR[wallR]*abs(R[5 ]-R[4 ])/abs(R[wallR]-R[wallR-1])))),16)

# Gradual coarsening
NR[ 4] = max(int(round(0.9*NR[wallR]*abs(R[4 ]-R[3 ])/abs(R[wallR]-R[wallR-1]))),4)
NR[ 3] = max(int(round(0.35*NR[wallR]*abs(R[3 ]-R[2 ])/abs(R[wallR]-R[wallR-1]))),2)
NR[ 2] = min(int(round(NR[2]*1.3)),max(int(round(NR[3]*abs(R[2 ]-R[1 ])/abs(R[3]-R[2]))),2))
# Now figure out grading of each block
for ir in range(len(R)):
    gradR_l.append(1.0)
    gradR_r.append(1.0)
    gradR.append(1.0)

#Outlet
Length = (R[wallR+1]-R[wallR])
deltaE = (R[wallR]-R[wallR-1])/NR[wallR]
#gradR_l[12]=1/bissection(Length/deltaE, fun, int(NR[12]))
if int(NR[wallR+1])>1:
   gradR[wallR+1]=1/bissection(Length/deltaE, fun, int(NR[wallR+1]))
#Block4
Length = (R[4]-R[3])
deltaE = (R[5]-R[4])/NR[5]
if int(NR[4])>1:
   gradR[4]=bissection(Length/deltaE, fun, int(NR[4]))
#Block3
Length = (R[3]-R[2])
deltaE = ((R[5]-R[4])/NR[5])*(1/gradR[4])
gradR[3]=bissection(Length/deltaE, fun, int(NR[3]))


#Block2
Length = (R[2]-R[1])*0.25
deltaE = (R[1]-R[0])/NR[1]
#gradR_l[2]=1/bissection(Length/deltaE, fun, int(0.25*NR[2]))
Length = (R[2]-R[1])*0.25
deltaE = ((R[5]-R[4])/NR[5])*(1/gradR[4])*(1/gradR[3])
#gradR_r[2]=bissection(Length/deltaE, fun, int(0.25*NR[2]))






NVert.append(int(round(100*Refinement)))
for i in range(len(L)-2):
    NVert.append(max(int(round(NVert[0]*abs(L[i+2]-L[i+1])/abs(L[1]-L[0]))),1))
NVert[4]=max(1,NVert[4])
if len(NVert)>5:
    NVert[5]=max(int(round(NVert[4]*(L[5]-L[6])/(L[4]-L[5]))),1)
if len(NVert)>6:
    NVert[6]=max(int(round(NVert[4]*(L[6]-L[7])/(L[4]-L[5]))),1)
if len(NVert)>7:
    NVert[7]=max(int(round(NVert[4]*(L[7]-L[8])/(L[4]-L[5]))),1)
if len(NVert)>8:
    NVert[8]=max(int(round(NVert[4]*(L[8]-L[9])/(L[4]-L[5]))),1)
if len(NVert)>9:
    NVert[9]=max(int(round(NVert[4]*(L[9]-L[10])/(L[4]-L[5]))),1)
if len(NVert)>10:
    NVert[10]=max(int(round(NVert[4]*(L[10]-L[11])/(L[4]-L[5]))),1)

for il in range(len(L)-1):
    gradVert.append(1.0)

# Refine near platter mesh and stretch
NVert[3]=int(NVert[3]*2.25)
Length = L[3]-L[4]
deltaE = (L[2]-L[3])/NVert[2]
if NVert[3]>1:
   gradVert[3]=(bissection(Length/deltaE, fun, NVert[3]))


# Uniform mesh after that
previousGridSize = (L[4]-L[5])/NVert[4]
newGridSize = (L[2]-L[3])*(1/gradVert[3])/NVert[2]
NVert[4] = int(NVert[4]*previousGridSize/newGridSize)



NVert[6] = int(NVert[6]*2.7)
NVert[7] = int(NVert[7]*4)
NVert[8] = int(NVert[8]/(0.95))


Length = L[8]-L[9]
deltaE = (L[7]-L[8])/NVert[7]
if NVert[8]>1:
   gradVert[8]=(bissection(Length/deltaE, fun, NVert[8]))

Length = L[6]-L[7]
deltaE = (L[7]-L[8])/NVert[7]
if NVert[6]>1:
   gradVert[6]=1/(bissection(Length/deltaE, fun, NVert[6]))

# Find NVert[5] such that grid size matches on both ends
Length = L[5]-L[6]
deltaE = gradVert[6]*deltaE
mismatchOpt = 1000
nz5Attempt = int(NVert[4]*(L[5]-L[6])/(L[4]-L[5]))+1
nz5Opt = int(NVert[4]*(L[5]-L[6])/(L[4]-L[5]))+1
targetSize = (L[4]-L[5])/NVert[4]
while nz5Attempt>1:
    grad=1/(bissection(Length/deltaE, fun, nz5Attempt))
    mismatch = abs(deltaE*grad-targetSize)
    if mismatch<mismatchOpt:
        nz5Opt = nz5Attempt
        mismatchOpt = mismatch
    if mismatch>mismatchOpt:
        break
    nz5Attempt -= 1

NVert[5] = nz5Opt
if NVert[5]>1:
   gradVert[5]=1/(bissection(Length/deltaE, fun, NVert[5]))


# ~~~~ Write species Dict
fw=open(outfile, 'w+')
# Write Header
fw.write('FoamFile\n')
fw.write('{\n')
fw.write('    version     2.0;\n')
fw.write('    format      ascii;\n')
fw.write('    class       dictionary;\n')
fw.write('    object      blockMeshDict;\n')
fw.write('}\n')
fw.write('\n')
#fw.write('convertToMeters 0.001;\n')
# Write all radii
counter=1
for rval in R:
    fw.write('R'+str(counter)+ ' ' + str(rval)+';\n')
    counter =  counter+1
fw.write('\n')
# Write all minus radii
counter=1
for rval in R:
    fw.write('mR'+str(counter)+ ' ' + str(-rval)+';\n')
    counter =  counter+1
fw.write('\n')
# Write all Length
counter=1
for lval in L:
    fw.write('L'+str(counter)+ ' ' + str(lval)+';\n')
    counter =  counter+1
fw.write('\n')
# Write all C
counter=1
for rval in R:
    fw.write('CW'+str(counter)+ ' ' + str(CW[counter-1])+';\n')
    fw.write('mCW'+str(counter)+ ' ' + str(mCW[counter-1])+';\n')
    fw.write('C1'+str(counter)+ ' ' + str(C1[counter-1])+';\n')
    fw.write('mC1'+str(counter)+ ' ' + str(mC1[counter-1])+';\n')
    fw.write('C2'+str(counter)+ ' ' + str(C2[counter-1])+';\n')
    fw.write('mC2'+str(counter)+ ' ' + str(mC2[counter-1])+';\n')
    fw.write('\n')
    counter =  counter+1
    
# Write all Ngrid
counter=1
for nR in NR:
    fw.write('NR'+str(counter)+ ' ' + str(NR[counter-1])+';\n')
    counter =  counter+1
fw.write('\n')
counter=1
for nS in NS:
    fw.write('NS'+str(counter)+ ' ' + str(NS[counter-1])+';\n')
    counter =  counter+1
fw.write('\n')
counter=1
for nVert in NVert:
    fw.write('NVert'+str(counter)+ ' ' + str(NVert[counter-1])+';\n')
    counter =  counter+1
fw.write('\n')



# ~~~~ Write vertices
fw.write('vertices\n')
fw.write('(\n')
# Write the squares first
counter = 0
for i in range(len(L)):
    fw.write('     ($mCW1  $mCW1  $L'+str(i+1)+')' + '  //'+str(counter)+'\n') 
    fw.write('     ( $CW1  $mCW1  $L'+str(i+1)+')\n') 
    fw.write('     ( $CW1   $CW1  $L'+str(i+1)+')\n') 
    fw.write('     ($mCW1   $CW1  $L'+str(i+1)+')\n') 
    fw.write('\n')
    counter=counter+4
# Write the circles then
for ir in range(len(R)):
    for il in range(len(L)):
        fw.write('    ($mC1'+str(ir+1)+'   $mC2'+str(ir+1)+'   $L'+str(il+1)+')' + '  //'+str(counter)+'\n')
        fw.write('    ( $C1'+str(ir+1)+'   $mC2'+str(ir+1)+'   $L'+str(il+1)+')\n')
        fw.write('    ( $C1'+str(ir+1)+'    $C2'+str(ir+1)+'   $L'+str(il+1)+')\n')
        fw.write('    ($mC1'+str(ir+1)+'    $C2'+str(ir+1)+'   $L'+str(il+1)+')\n')
        fw.write('\n')
        counter=counter+4
fw.write(');\n')
fw.write('\n')

# ~~~~ Write blocks
fw.write('blocks\n')
fw.write('(\n')
# Write the squares first
for i in range(N2):
    gradingZ = gradVert[i]       
    # Am I a wall
    iwall = amIwall(WallL,WallR,-1,i)
    if iwall==1:
        fw.write('//')
    
    i1 = int((i+1)*4)
    i2 = int(i*4)
    fw.write('     hex ('+ str(i1)+' '+str(i1+1) + ' ' + str(i1+2) + ' ' + str(i1+3))
    fw.write(' '         + str(i2)+' '+str(i2+1) + ' ' + str(i2+2) + ' ' + str(i2+3)+')')
    fw.write(' ($NS1 $NS1 $NVert'+str(i+1)+')' + ' simpleGrading (1 1 '+ str(gradingZ)+')\n')
fw.write('\n')
# Write the squares then
for ir in range(N1):
    for il in range(N2):

        #gradingZ = 1
        #gradingR = 1
        #if il==N2-1:
        #    gradingZ = outletGrading
        gradingR_l = gradR_l[ir]       
        gradingR_r = gradR_r[ir]       
        gradingZ = gradVert[il]       
        gradingR = gradR[ir]
        # Am I a wall
        iwall = amIwall(WallL,WallR,ir,il)
        # bottom right corner
        i1 = int(4*(N2+1)*ir + 4*(il+1))
        # bottom left corner
        i2 = int(4*(N2+1)*(ir+1) + 4*(il+1))
        # top right corner
        i3 = int(4*(N2+1)*ir + 4*(il))
        # top left corner
        i4 = int(4*(N2+1)*(ir+1) + 4*(il))
        #outlet
        if iwall==1:
            fw.write('//')
        fw.write('     hex ('+str(i1)+ ' ' + str(i2) + ' ' + str(i2+1) + ' ' + str(i1+1) + ' ')
        fw.write(            str(i3) + ' ' + str(i4) + ' ' + str(i4+1) + ' ' + str(i3+1) +') ')
        fw.write(' ($NR'+str(ir+1)+' $NS1'+' $NVert' + str(il+1) + ') simpleGrading ('+str(gradingR)+' 1 '+str(gradingZ)+')\n')
        if iwall==1:
            fw.write('//')
        fw.write('     hex ('+str(i1+1)+ ' ' + str(i2+1) + ' ' + str(i2+2) + ' ' + str(i1+2) + ' ')
        fw.write(            str(i3+1) + ' ' + str(i4+1) + ' ' + str(i4+2) + ' ' + str(i3+2) +') ')
        fw.write(' ($NR'+str(ir+1)+' $NS1'+' $NVert' + str(il+1) + ') simpleGrading ('+str(gradingR)+' 1 '+str(gradingZ)+')\n')

        if iwall==1:
            fw.write('//')
        fw.write('     hex ('+str(i1+2)+ ' ' + str(i2+2) + ' ' + str(i2+3) + ' ' + str(i1+3) + ' ')
        fw.write(            str(i3+2) + ' ' + str(i4+2) + ' ' + str(i4+3) + ' ' + str(i3+3) +') ')
        fw.write(' ($NR'+str(ir+1)+' $NS1'+' $NVert' + str(il+1) + ') simpleGrading ('+str(gradingR)+' 1 '+str(gradingZ)+')\n')

        if iwall==1:
            fw.write('//')
        fw.write('     hex ('+str(i1+3)+ ' ' + str(i2+3) + ' ' + str(i2) + ' ' + str(i1) + ' ')
        fw.write(            str(i3+3) + ' ' + str(i4+3) + ' ' + str(i4) + ' ' + str(i3) +') ')
        fw.write(' ($NR'+str(ir+1)+' $NS1'+' $NVert' + str(il+1) + ') simpleGrading ('+str(gradingR)+' 1 '+str(gradingZ)+')\n')
        fw.write('\n')
fw.write(');\n')
fw.write('\n')


# ~~~~ Write edges
fw.write('edges\n')
fw.write('(\n')
ind=len(L)*4
for ir in range(len(R)):
    for il in range(len(L)):
        
        
        # Edges should be removed if they are surrounded by walls
        iwall1 = amIwall(WallL,WallR,ir,il)
        iwall2 = amIwall(WallL,WallR,ir+1,il)
        iwall3 = amIwall(WallL,WallR,ir,il-1)
        iwall4 = amIwall(WallL,WallR,ir+1,il-1)
        sumwall = iwall1+iwall2+iwall3+iwall4
        #if ind==68:
        #   print(il)
        #   print(ir)
        #   print(iwall1)
        #   print(iwall2)
        #   print(iwall3)
        #   print(iwall4)
        #   #stop
        comment=0
        if sumwall==4 or (il==0 and sumwall==2) or (il==len(L)-1 and sumwall==2) or (ir==len(R)-1 and sumwall==2) or (ir==len(R)-1 and il==0 and sumwall==1):
            comment=1

        if comment==1:
            fw.write('//')
        fw.write('    arc '+ str(ind) + ' ' + str(ind+1) +   ' (0' + '    $mR' +str(ir+1)+ '   $L'+str(il+1)+')\n')
        if comment==1:
            fw.write('//')
        fw.write('    arc '+ str(ind+1) + ' ' + str(ind+2) + ' ($R'+str(ir+1)+ '  0      $L'+str(il+1)+')\n')
        if comment==1:
            fw.write('//')
        fw.write('    arc '+ str(ind+2) + ' ' + str(ind+3) + ' (0' + '    $R' +str(ir+1)+ '    $L'+str(il+1)+')\n')
        if comment==1:
            fw.write('//')
        fw.write('    arc '+ str(ind+3) + ' ' + str(ind) +   ' ($mR'+str(ir+1)+ ' 0      $L'+str(il+1)+')\n')
        ind = ind+4
        fw.write('\n')
fw.write(');\n')
fw.write('\n')

# ~~~~ Write boundary
fw.write('boundary\n')
fw.write('(\n')

for i in range(len(BoundaryNames)):
    fw.write('    '+BoundaryNames[i]+'\n')
    fw.write('    '+'{\n')
    fw.write('        '+'type patch;\n')
    fw.write('        '+'faces\n')
    fw.write('        '+'(\n')
    
    for ibound in range(len(BoundaryType[i])):
        boundType = BoundaryType[i][ibound]
        if boundType=='lateral':
            rminInd = BoundaryRmin[i][ibound]
            rmaxInd = BoundaryRmax[i][ibound]
            lInd = BoundaryLmin[i][ibound]
            i1=rminInd*(4*(N2+1)) + 4*lInd #bottom
            i2=i1-4 #top
            fw.write('            '+'( ' + str(i1)   + ' ' + str(i1+1) + ' ' + str(i2+1) + ' ' +str(i2)   + ')\n')
            fw.write('            '+'( ' + str(i1+1) + ' ' + str(i1+2) + ' ' + str(i2+2) + ' ' +str(i2+1) + ')\n')
            fw.write('            '+'( ' + str(i1+2) + ' ' + str(i1+3) + ' ' + str(i2+3) + ' ' +str(i2+2) + ')\n')
            fw.write('            '+'( ' + str(i1+3) + ' ' + str(i1)   + ' ' + str(i2)   + ' ' +str(i2+3) + ')\n')
            fw.write('\n')

        elif boundType=='top':
            lminInd = BoundaryLmin[i][ibound]
            lmaxInd = BoundaryLmax[i][ibound]
            rInd = BoundaryRmin[i][ibound]  
            if rInd>0:
                i1 = 4*(N2+1)*(rInd-1) + 4*lminInd# right
                i2 = i1+4*(N2+1) # left
                fw.write('            '+'( ' + str(i1)   + ' ' + str(i2)   + ' ' + str(i2+1) + ' ' +str(i1+1) + ')\n')
                fw.write('            '+'( ' + str(i1+1) + ' ' + str(i2+1) + ' ' + str(i1+2) + ' ' +str(i2+2) + ')\n')
                fw.write('            '+'( ' + str(i1+2) + ' ' + str(i2+2) + ' ' + str(i2+3) + ' ' +str(i1+3) + ')\n')
                fw.write('            '+'( ' + str(i1+3) + ' ' + str(i2+3) + ' ' + str(i2)   + ' ' +str(i1)    + ')\n')
            else:
                i1 = lminInd*4
                fw.write('            '+'( ' + str(i1)   + ' ' + str(i1+1)   + ' ' + str(i1+2) + ' ' +str(i1+3) + ')\n')
            fw.write('\n')

        elif boundType=='bottom':
            lminInd = BoundaryLmin[i][ibound]
            lmaxInd = BoundaryLmax[i][ibound]
            rInd = BoundaryRmin[i][ibound]
            i1 = 4*(N2+1)*(rInd-1) + 4*lminInd# right
            i2 = i1+4*(N2+1) # left
            if rInd>0:
                fw.write('            '+'( ' + str(i1)   + ' ' + str(i2)   + ' ' + str(i2+1) + ' ' +str(i1+1) + ')\n')
                fw.write('            '+'( ' + str(i1+1) + ' ' + str(i2+1) + ' ' + str(i1+2) + ' ' +str(i2+2) + ')\n')
                fw.write('            '+'( ' + str(i1+2) + ' ' + str(i2+2) + ' ' + str(i2+3) + ' ' +str(i1+3) + ')\n')
                fw.write('            '+'( ' + str(i1+3) + ' ' + str(i2+3) + ' ' + str(i2)   + ' ' +str(i1)    + ')\n')
            else:
                i1 = lminInd*4
                fw.write('            '+'( ' + str(i1)   + ' ' + str(i1+1)   + ' ' + str(i1+2) + ' ' +str(i1+3) + ')\n')
            fw.write('\n')
            

    fw.write('        '+');\n')
    fw.write('    '+'}\n')

fw.write(');\n')
fw.write('\n')



# ~~~~ Write mergePatchPairs
fw.write('mergePatchPairs\n')
fw.write('(\n')
fw.write(');\n')


fw.close()

















