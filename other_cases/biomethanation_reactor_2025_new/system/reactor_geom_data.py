import numpy as np

#geometry ========
conv_mtr=0.025 #in inches
Dt = 5.0              # Tank Diameter
Da = 2.75              # impeller tip Diameter
H = 62.854             # height of reactor (includes D/4 with the air phase only)
nimpellers = 8
Clocs = [3.44,4.18,8.86,8.86,5.89,5.89,6.072,6.072]
C=np.cumsum(Clocs).tolist()
W = 0.4   # impeller blade width 
L = 0.5   # # impeller blade length
Dh =1.75  # Hub Diameter
Lin = 0.05 # impeller blade length (inside the hub)
J =  0.5  # Baffle Width
Wh = 0.12  # Hub Width 
polyrad=0.625/2  # Stem radius (R_shaft)

Z0 = 0.0               # bottom of reactor
Dmrf = (Da+Dt-2*J)/2   # MRF region Diameter

#mesh ========
nr  = 5 #120	      # mesh points per unit radial length
nz  = 5 #240             # mesh points per unit axial length
Npoly = 4             # mesh points in the polygon at the axis
Na = 6               # mesh points in the azimuthal direction

nbaffles = 6          # number of baffles and impeller fins

nsplits=2*nbaffles    #we need twice the number of splits
dangle=2.0*np.pi/float(nsplits)

bladepitch=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

#curved bottom params
use_curved=0
cb_center=[0.0,0.0,Z0+100*H]
edgep=[Dt/2,0.0,Z0]
cb_rad=np.sqrt((edgep[0]-cb_center[0])**2+(edgep[1]-cb_center[1])**2+(edgep[2]-cb_center[2])**2)

circradii=np.array([Dh/2-Lin,Dh/2,Da/2,Dmrf/2,Dt/2-J,Dt/2])
ncirc = len(circradii)
hub_circ   = 1 
inhub_circ = hub_circ-1  #circle inside hub
rot_circ   = hub_circ+1
mrf_circ   = rot_circ+1 
tank_circ = ncirc-1 

reacthts = [Z0]
baff_sections = []
baff_volumes = []
hub_volumes=[]
count=1
angle_offsets=[0.0]
for n_imp in range(nimpellers):
    reacthts.append(Z0 + C[n_imp] -  W/2)

    baff_sections.append(count)
    baff_volumes.append(count)
    angle_offsets.append(-bladepitch[n_imp])
    count=count+1

    reacthts.append(Z0 + C[n_imp] - Wh/2)

    baff_sections.append(count)
    baff_volumes.append(count)
    hub_volumes.append(count)
    angle_offsets.append(0.0)
    count=count+1

    reacthts.append(Z0 + C[n_imp] + Wh/2)

    baff_sections.append(count)
    baff_volumes.append(count)
    angle_offsets.append(0.0)
    count=count+1

    reacthts.append(Z0 + C[n_imp] +  W/2)
    baff_sections.append(count)
    angle_offsets.append(bladepitch[n_imp])
    count=count+1

reacthts.append(Z0+H)
angle_offsets.append(0.0)


nsections = len(reacthts)
nvolumes = nsections-1
meshz = nz*np.diff(reacthts)
meshz = meshz.astype(int)+1 #avoid zero mesh elements

all_volumes=range(nvolumes)
nonbaff_volumes=[sec for sec in all_volumes if sec not in baff_volumes]
nonstem_volumes=[] #this is 0,1 no matter how many impellers are there


#note: stem_volumes include hub volumes also
#these are volumes where we miss out polygon block
stem_volumes=[sec for sec in all_volumes if sec not in nonstem_volumes]

#removes hub_volumes here for declaring patches
only_stem_volumes=[sec for sec in stem_volumes if sec not in hub_volumes]
#print(only_stem_volumes)
#print(stem_volumes)
#print(hub_volumes)
#print(all_volumes)
#to define mrf region
#not that [1] is not a stem volume but baffles are there
#mrf_volumes=[1]+stem_volumes
mrf_volumes=stem_volumes

#increase grid points in the impeller section
for i in baff_volumes:
    meshz[i] *=2 

meshr = nr*np.diff(circradii)

#adding polygon to hub mesh resolution
meshr = np.append(nr*polyrad,meshr)
meshr = meshr.astype(int)
meshr += 1 # to avoid being zero

centeroffset     = 1 #one point on the axis
polyoffset       = nsplits #number of points on polygon
npts_per_section = centeroffset + polyoffset + ncirc*nsplits #center+polygon+circles
