import numpy as np

#geometry ========
T = 0.525*2
Dt = T               # Tank Diameter
Da = T/3              # impeller tip Diameter
H = T*2.0             # height of reactor (includes D/4 with the air phase only)
nimpellers = 2 # NOT SURE
#S_i = 0.381 # 15" impeller spacing 
C = [0.47*T/2,1.62*T/2]
#C = [D1.01*S_i,2.0*S_i,3.0*S_i,4.0*S_i,5.0*S_i,6.0*S_i,7.0*S_i,8.0*S_i,9.0*S_i,10.0*S_i]      # height of the center of impellers
W = T/10   # NOT SURE estimate            # impeller blade width 
L = T/20   # NOT SURE estimate            # impeller blade length (beyond the hub) W=Da/5, L=Da/4
Dh =Da-2*L # NOT SURE            # Hub Diameter
Lin = L    # NOT SURE            # impeller blade length (inside the hub)
J =  T/10              # Baffle Width
Wh = W/10  #NOT SURE             # Hub height (Width) 
polyrad=T/30  #NOT SURE       # Stem radius (R_shaft)

Z0 = 0.0               # bottom of reactor
Dmrf = (Da+Dt-2*J)/2   # MRF region Diameter

#mesh ========
nr  = 120 #120	      # mesh points per unit radial length
nz  = 240 #240             # mesh points per unit axial length
Npoly = 4             # mesh points in the polygon at the axis
Na = 6               # mesh points in the azimuthal direction

nbaffles = 6          # number of baffles and impeller fins

nsplits=2*nbaffles    #we need twice the number of splits
dangle=2.0*np.pi/float(nsplits)

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
for n_imp in range(nimpellers):
    reacthts.append(Z0 + C[n_imp] -  W/2)

    baff_sections.append(count)
    baff_volumes.append(count)
    count=count+1

    reacthts.append(Z0 + C[n_imp] - Wh/2)

    baff_sections.append(count)
    baff_volumes.append(count)
    hub_volumes.append(count)
    count=count+1

    reacthts.append(Z0 + C[n_imp] + Wh/2)

    baff_sections.append(count)
    baff_volumes.append(count)
    count=count+1

    reacthts.append(Z0 + C[n_imp] +  W/2)
    baff_sections.append(count)
    count=count+1

reacthts.append(Z0+H)


nsections = len(reacthts)
nvolumes = nsections-1
meshz = nz*np.diff(reacthts)
meshz = meshz.astype(int)+1 #avoid zero mesh elements

all_volumes=range(nvolumes)
nonbaff_volumes=[sec for sec in all_volumes if sec not in baff_volumes]
nonstem_volumes=[0,1] #this is 0,1 no matter how many impellers are there


#note: stem_volumes include hub volumes also
#these are volumes where we miss out polygon block
stem_volumes=[sec for sec in all_volumes if sec not in nonstem_volumes]

#removes hub_volumes here for declaring patches
only_stem_volumes=[sec for sec in stem_volumes if sec not in hub_volumes]

#to define mrf region
#not that [1] is not a stem volume but baffles are there
mrf_volumes=[1]+stem_volumes

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
