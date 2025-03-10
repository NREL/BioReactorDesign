import numpy as np
from reactor_geom_data import *

def write_ofoam_preamble(outfile):

    outfile.write("/*--------------------------------*- C++ -*----------------------------------*\\\n")
    outfile.write("| =========                 |                                                 |\n")
    outfile.write("| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n")
    outfile.write("|  \\    /   O peration     | Version:  5                                     |\n")
    outfile.write("|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |\n")
    outfile.write("|    \\/     M anipulation  |                                                 |\n")
    outfile.write("\*---------------------------------------------------------------------------*/\n")
    outfile.write("FoamFile\n")
    outfile.write("{\n")
    outfile.write("\tversion     2.0;\n")
    outfile.write("\tformat      ascii;\n")
    outfile.write("\tclass       dictionary;\n")
    outfile.write("\tobject      blockMeshDict;\n")
    outfile.write("}\n\n")
    outfile.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n")
    outfile.write("convertToMeters %f;\n\n"%(conv_mtr))
    outfile.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n")

def write_vertices(outfile):

    outfile.write("\n// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n")
    outfile.write("vertices\n(\n")
    counter=0
    for repeat in range(2):
        
        outfile.write("\n//section "+str(0)+"\n")
        outfile.write("\n//center\n")
        outfile.write("(0.0 0.0 "+str(cb_center[2]-cb_rad)+") // "+str(counter)+"\n")
        counter=counter+1

        #polygon section
        outfile.write("\n//polygon\n")
        for i in range(nsplits):
            ang = i*dangle
            azang=np.pi/2-np.arccos(polyrad/cb_rad)
            x   = polyrad*np.cos(ang) 
            y   = polyrad*np.sin(ang)
            outfile.write("( "+str(x)+" "+str(y)+" "+str(cb_center[2]-cb_rad*np.cos(azang))+" ) // "+str(counter)+"\n") 
            counter=counter+1

            outfile.write("\n//circles\n")

        for ci in range(ncirc):
            for i in range(nsplits):
                ang = i*dangle
                azang=np.pi/2-np.arccos(circradii[ci]/cb_rad)
                x   = circradii[ci]*np.cos(ang)
                y   = circradii[ci]*np.sin(ang)
                outfile.write("( "+str(x)+" "+str(y)+" "+str(cb_center[2]-cb_rad*np.cos(azang))+" ) //"+str(counter)+"\n")  
                counter=counter+1

        for zi in range(1,nsections):

            outfile.write("\n//section "+str(zi)+"\n")
            outfile.write("\n//center\n")
            outfile.write("(0.0 0.0 "+str(reacthts[zi])+") // "+str(counter)+"\n")
            counter=counter+1

            #polygon section
            outfile.write("\n//polygon\n")
            for i in range(nsplits):
                ang = i*dangle
                x   = polyrad*np.cos(ang) 
                y   = polyrad*np.sin(ang)
                outfile.write("( "+str(x)+" "+str(y)+" "+str(reacthts[zi])+" ) // "+str(counter)+"\n") 
                counter=counter+1

            outfile.write("\n//circles\n")

            for ci in range(ncirc):

                outfile.write("\n//circle "+str(ci)+"\n")
                offsetang=0.0
                if(ci>=0 and ci<=2):
                    offsetang=angle_offsets[zi]

                for i in range(nsplits):
                    ang = i*dangle+offsetang
                    x   = circradii[ci]*np.cos(ang)
                    y   = circradii[ci]*np.sin(ang)
                    outfile.write("( "+str(x)+" "+str(y)+" "+str(reacthts[zi])+" ) //"+str(counter)+"\n")  
                    counter=counter+1


    outfile.write(");\n")

def get_globalindex_of(splti,ci,zi):

    #also works for ci=-1
    global_id   =  zi*npts_per_section + centeroffset + polyoffset + ci*nsplits + splti%nsplits
    return(global_id)

def get_baffle_point_of(splti,ci,zi):


    baffle_id = get_globalindex_of(splti,ci,zi)

    if(zi in baff_sections):
        
        if(ci==hub_circ):

            if(splti%2==0): #even number for impeller
                baffle_id += nsections*npts_per_section

        if(ci==tank_circ):

            if(splti%2==1): #odd number for baffle
                baffle_id += nsections*npts_per_section

    return(baffle_id)

def write_edges(outfile):

    outfile.write("\n// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n")
    outfile.write("edges\n(\n")
    outfile.write("\n//circles\n")
    
    outfile.write("\n//section "+str(0)+"\n")
    
    #polygon section
    outfile.write("\n//polygon\n")
    for i in range(nsplits):
        ang = i*dangle
        nextang=(i+1)*dangle
        midx   = 0.5*polyrad*(np.cos(ang)+np.cos(nextang))
        midy   = 0.5*polyrad*(np.sin(ang)+np.sin(nextang))
        azang  = np.pi/2-np.arccos(np.sqrt(midx**2+midy**2)/cb_rad)
        outfile.write("arc "+str(i+1)+" "+str((i+1)%nsplits+1)+" ")
        outfile.write("( "+str(midx)+" "+str(midy)+" "+str(cb_center[2]-cb_rad*np.cos(azang))+" )\n") 

    for ci in range(ncirc):
        outfile.write("\n//circle "+str(ci)+"\n")
        for i in range(nsplits):
            ang = i*dangle
            midx = circradii[ci]*np.cos(ang+dangle/2)
            midy = circradii[ci]*np.sin(ang+dangle/2)
            azang= np.pi/2-np.arccos(circradii[ci]/cb_rad)

            globalind1 = get_baffle_point_of(i,ci,0)
            globalind2 = get_globalindex_of(i+1,ci,0)

            outfile.write("arc "+str(globalind1)+" "+str(globalind2)+" ")
            outfile.write("( "+str(midx)+" "+str(midy)+" "+str(cb_center[2]-cb_rad*np.cos(azang))+" )\n")  
            
    for zi in range(1,nsections):

        outfile.write("\n//section "+str(zi)+"\n")

        offset=1+nsplits #one for center and nsplits for polygon

        outfile.write("\n//circles\n")
        for ci in range(ncirc):
            outfile.write("\n//circle "+str(ci)+"\n")
            for i in range(nsplits):
                ang = i*dangle
                midx = circradii[ci]*np.cos(ang+dangle/2)
                midy = circradii[ci]*np.sin(ang+dangle/2)

                globalind1 = get_baffle_point_of(i,ci,zi)
                globalind2 = get_globalindex_of(i+1,ci,zi)

                outfile.write("arc "+str(globalind1)+" "+str(globalind2)+" ")
                outfile.write("( "+str(midx)+" "+str(midy)+" "+str(reacthts[zi])+" )\n")  


    outfile.write(");\n")

def write_this_block(outfile,comment,ids,mesh,zonename="none"):
    
    outfile.write("\n //"+comment+"\n")
    outfile.write("hex (")
    for i in range(len(ids)):
        outfile.write(str(ids[i])+" ")
    outfile.write(")\n")

    if(zonename != "none"):
        outfile.write(zonename+"\n")

    outfile.write("( %d %d %d )\n"%(mesh[0],mesh[1],mesh[2]))
    outfile.write("SimpleGrading (1 1 1)\n")


def write_blocks(outfile):

    outfile.write("\n// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n")
    outfile.write("blocks\n(\n")

    idarray   = np.zeros(8,dtype=int)
    mesharray = np.zeros(3,dtype=int)


    for zi in range(nvolumes):

        
        outfile.write("\n//section "+str(zi)+"-"+str(zi+1)+"\n")

        offset0=zi*npts_per_section
        offset1=(zi+1)*npts_per_section

        #polygon section
        center_id0 = offset0
        center_id1 = offset1

        #skip polygon blocks in stem sections 
        if(zi in nonstem_volumes): 
            for i in range(nsplits):

                localind1=centeroffset+i%nsplits
                localind2=centeroffset+(i+1)%nsplits

                idarray[0] = offset0+localind1
                idarray[1] = offset1+localind1
                idarray[2] = offset1+localind2
                idarray[3] = offset0+localind2
                idarray[4] = center_id0
                idarray[5] = center_id1
                idarray[6] = center_id1
                idarray[7] = center_id0

                mesharray[0] = meshz[zi]
                mesharray[1] = Na
                mesharray[2] = Npoly

                zonename="none"
                if((zi in mrf_volumes)):
                    zonename="rotor"

                write_this_block(outfile,"block %d"%(i),idarray,mesharray,zonename)
    

        idarray[:]=0
        mesharray[:]=0
        outfile.write("\n//circles\n")

        for ci in range(ncirc):

            zonename="none"

            #skip blocks inside hub
            if( ((ci==inhub_circ) or (ci==hub_circ)) and (zi in hub_volumes)):
                continue

            outfile.write("\n//circle "+str(ci)+"\n")

            if((zi in mrf_volumes) and (ci<=mrf_circ)):
                zonename="rotor"

            for i in range(nsplits):

                idarray[0] = get_baffle_point_of(i,ci,zi)
                idarray[1] = get_baffle_point_of(i,ci,zi+1)
                idarray[2] = get_globalindex_of(i+1,ci,zi+1)
                idarray[3] = get_globalindex_of(i+1,ci,zi)
                idarray[4] = get_baffle_point_of(i,ci-1,zi)
                idarray[5] = get_baffle_point_of(i,ci-1,zi+1)
                idarray[6] = get_globalindex_of(i+1,ci-1,zi+1)
                idarray[7] = get_globalindex_of(i+1,ci-1,zi)

                mesharray[0] = meshz[zi]
                mesharray[1] = Na
                mesharray[2] = meshr[ci]
                write_this_block(outfile,"block %d"%(i),idarray,mesharray,zonename)


    outfile.write(");\n")

def write_patches(outfile):

    inhub_ci = inhub_circ
    hub_ci   = hub_circ
    rot_ci   = rot_circ
    poly_ci  = -1

    outfile.write("\n// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n")
    outfile.write("patches\n(\n")

    #inlet patch
    #zi=0
    #centerid=zi*npts_per_section

    #outfile.write("\n\tpatch inlet\n\t(\n")

    #polygon
    #outfile.write("\n\t\t//polygon\n")
    #for i in range(nsplits):
        #outfile.write("\t\t( ")
        #outfile.write(str(get_globalindex_of(i,poly_ci,zi))+" ")
        #outfile.write(str(get_globalindex_of(i+1,poly_ci,zi))+" ")
        #outfile.write(str(centerid)+" ")
        #outfile.write(str(centerid)+")\n")
    
    '''outfile.write("\n\t\t//inhub_circ to polygon\n")
    for i in range(nsplits):
        outfile.write("\t\t( ")
        outfile.write(str(get_globalindex_of(i,inhub_ci,zi))+" ")
        outfile.write(str(get_globalindex_of(i+1,inhub_ci,zi))+" ")
        outfile.write(str(get_globalindex_of(i+1,poly_ci,zi))+" ")
        outfile.write(str(get_globalindex_of(i,poly_ci,zi))+")\n")

    outfile.write("\n\t\t//hub to inhub_circ\n")
    for i in range(nsplits):
        outfile.write("\t\t( ")
        outfile.write(str(get_globalindex_of(i,hub_ci,zi))+" ")
        outfile.write(str(get_globalindex_of(i+1,hub_ci,zi))+" ")
        outfile.write(str(get_globalindex_of(i+1,inhub_ci,zi))+" ")
        outfile.write(str(get_globalindex_of(i,inhub_ci,zi))+")\n")

    outfile.write("\n\t\t//rotor to hub\n")
    for i in range(nsplits):
        outfile.write("\t\t( ")
        outfile.write(str(get_globalindex_of(i,rot_ci,zi))+" ")
        outfile.write(str(get_globalindex_of(i+1,rot_ci,zi))+" ")
        outfile.write(str(get_globalindex_of(i+1,hub_ci,zi))+" ")
        outfile.write(str(get_globalindex_of(i,hub_ci,zi))+")\n")'''

    #outfile.write("\t)\n")
    
    #outlet patch
    zi=nsections-1
    centerid=zi*npts_per_section
    outfile.write("\n\tpatch outlet\n\t(\n")

    #no polygon patch in outlet when we include stem
    #polygon
    #outfile.write("\n\t\t//polygon\n")
    #for i in range(nsplits):
    #    outfile.write("\t\t( ")
    #    outfile.write(str(get_globalindex_of(i,poly_ci,zi))+" ")
    #    outfile.write(str(get_globalindex_of(i+1,poly_ci,zi))+" ")
    #    outfile.write(str(centerid)+" ")
    #    outfile.write(str(centerid)+")\n")

    outfile.write("\n\t\t//circles\n")
    for ci in range(ncirc):
        outfile.write("\n\t\t//circle "+str(ci)+" - "+str(ci-1)+" \n")
        for i in range(nsplits):
            outfile.write("\t\t( ")
            outfile.write(str(get_globalindex_of(i,ci,zi))+" ")
            outfile.write(str(get_globalindex_of(i+1,ci,zi))+" ")
            outfile.write(str(get_globalindex_of(i+1,ci-1,zi))+" ")
            outfile.write(str(get_globalindex_of(i,ci-1,zi))+")\n")

    outfile.write("\t)\n")

    #propeller patch
    outfile.write("\n\twall propeller\n\t(\n")
    
    #need polygon patch at the first impeller
    #zi=hub_volumes[0]
    #outfile.write("\n\t\t//polygon\n")
    #centerid=zi*npts_per_section
    #polygon
    #for i in range(nsplits):
    #    outfile.write("\t\t( ")
    #    outfile.write(str(get_globalindex_of(i,poly_ci,zi))+" ")
    #    outfile.write(str(get_globalindex_of(i+1,poly_ci,zi))+" ")
    #    outfile.write(str(centerid)+" ")
    #    outfile.write(str(centerid)+")\n")

    for n_imp in range(nimpellers):
        zi_bottom = hub_volumes[n_imp]        #bottom of impeller section
        zi_top    = zi_bottom+1               #bottom of impeller section

        for zi in [zi_bottom,zi_top]:


            outfile.write("\n\t\t//hub to blade circle\n")
            for i in range(nsplits):
                outfile.write("\t\t( ")
                outfile.write(str(get_baffle_point_of(i,hub_ci,zi))+" ")
                outfile.write(str(get_globalindex_of(i+1,hub_ci,zi))+" ")
                outfile.write(str(get_globalindex_of(i+1,inhub_ci,zi))+" ")
                outfile.write(str(get_baffle_point_of(i,inhub_ci,zi))+")\n")

            outfile.write("\n\t\t//blade to polygon\n")
            for i in range(nsplits):
                outfile.write("\t\t( ")
                outfile.write(str(get_baffle_point_of(i,inhub_ci,zi))+" ")
                outfile.write(str(get_globalindex_of(i+1,inhub_ci,zi))+" ")
                outfile.write(str(get_globalindex_of(i+1,poly_ci,zi))+" ")
                outfile.write(str(get_baffle_point_of(i,poly_ci,zi))+")\n")

        #sides
        outfile.write("\n\t\t//sides\n")
        for i in range(nsplits):
            outfile.write("\t\t( ")
            outfile.write(str(get_baffle_point_of(i,hub_ci,zi_bottom))+" ")
            outfile.write(str(get_globalindex_of(i+1,hub_ci,zi_bottom))+" ")
            outfile.write(str(get_globalindex_of(i+1,hub_ci,zi_top))+" ")
            outfile.write(str(get_baffle_point_of(i,hub_ci,zi_top))+")\n")

    #blades
    outfile.write("\n\t\t//blades\n")
    for zi in baff_volumes:
        zi_bottom = zi
        zi_top    = zi+1
        for i in range(0,nsplits,2): #even numbers
            outfile.write("\t\t( ")
            outfile.write(str(get_baffle_point_of(i,hub_ci+1,zi_bottom))+" ")
            outfile.write(str(get_baffle_point_of(i,hub_ci+1,zi_top))+" ")
            outfile.write(str(get_baffle_point_of(i,hub_ci,zi_top))+" ")
            outfile.write(str(get_baffle_point_of(i,hub_ci,zi_bottom))+")\n")

            outfile.write("\t\t( ")
            outfile.write(str(get_globalindex_of(i,hub_ci+1,zi_bottom))+" ")
            outfile.write(str(get_globalindex_of(i,hub_ci+1,zi_top))+" ")
            outfile.write(str(get_globalindex_of(i,hub_ci,zi_top))+" ")
            outfile.write(str(get_globalindex_of(i,hub_ci,zi_bottom))+")\n")
    

    #inside hub blades
    for n_imp in range(nimpellers):
        hub_vol=hub_volumes[n_imp]
        zi_pairs=[[hub_vol-1,hub_vol],[hub_vol+1,hub_vol+2]]

        for zi_pair in zi_pairs:

            zi_below=zi_pair[0]
            zi_above=zi_pair[1]

            for i in range(0,nsplits,2): #even numbers
                outfile.write("\t\t( ")
                outfile.write(str(get_baffle_point_of(i,hub_ci,zi_below))+" ")
                outfile.write(str(get_baffle_point_of(i,inhub_ci,zi_below))+" ")
                outfile.write(str(get_baffle_point_of(i,inhub_ci,zi_above))+" ")
                outfile.write(str(get_baffle_point_of(i,hub_ci,zi_above))+")\n")

                outfile.write("\t\t( ")
                outfile.write(str(get_globalindex_of(i,hub_ci,zi_below))+" ")
                outfile.write(str(get_globalindex_of(i,inhub_ci,zi_below))+" ")
                outfile.write(str(get_globalindex_of(i,inhub_ci,zi_above))+" ")
                outfile.write(str(get_globalindex_of(i,hub_ci,zi_above))+")\n")


    #stem
    outfile.write("\n\t\t//stem sides\n")
    for zi in only_stem_volumes:
        zi_bottom = zi
        zi_top    = zi+1
        for i in range(nsplits):
            outfile.write("\t\t( ")
            outfile.write(str(get_globalindex_of(i,poly_ci,zi_bottom))+" ")
            outfile.write(str(get_globalindex_of(i+1,poly_ci,zi_bottom))+" ")
            outfile.write(str(get_globalindex_of(i+1,poly_ci,zi_top))+" ")
            outfile.write(str(get_globalindex_of(i,poly_ci,zi_top))+")\n")
    
    outfile.write("\t)\n")

    #stator and walls patch
    outfile.write("\n\twall walls\n\t(\n")
    tank_ci=ncirc-1
    
    #inlet section now in wall=====================================
    zi=0
    centerid=zi*npts_per_section
    outfile.write("\n\t\t//inhub_circ to polygon\n")
    for i in range(nsplits):
        outfile.write("\t\t( ")
        outfile.write(str(get_globalindex_of(i,inhub_ci,zi))+" ")
        outfile.write(str(get_globalindex_of(i+1,inhub_ci,zi))+" ")
        outfile.write(str(get_globalindex_of(i+1,poly_ci,zi))+" ")
        outfile.write(str(get_globalindex_of(i,poly_ci,zi))+")\n")

    outfile.write("\n\t\t//hub to inhub_circ\n")
    for i in range(nsplits):
        outfile.write("\t\t( ")
        outfile.write(str(get_globalindex_of(i,hub_ci,zi))+" ")
        outfile.write(str(get_globalindex_of(i+1,hub_ci,zi))+" ")
        outfile.write(str(get_globalindex_of(i+1,inhub_ci,zi))+" ")
        outfile.write(str(get_globalindex_of(i,inhub_ci,zi))+")\n")

    outfile.write("\n\t\t//rotor to hub\n")
    for i in range(nsplits):
        outfile.write("\t\t( ")
        outfile.write(str(get_globalindex_of(i,rot_ci,zi))+" ")
        outfile.write(str(get_globalindex_of(i+1,rot_ci,zi))+" ")
        outfile.write(str(get_globalindex_of(i+1,hub_ci,zi))+" ")
        outfile.write(str(get_globalindex_of(i,hub_ci,zi))+")\n")
    #==============================================================

    for zi in range(nsections-1):
        outfile.write("\n\t\t//tank walls "+str(zi)+" - "+str(zi+1)+"\n")

        for i in range(nsplits):
            outfile.write("\t\t( ")
            outfile.write(str(get_baffle_point_of(i,tank_ci,zi))+" ")
            outfile.write(str(get_globalindex_of(i+1,tank_ci,zi))+" ")
            outfile.write(str(get_globalindex_of(i+1,tank_ci,zi+1))+" ")
            outfile.write(str(get_baffle_point_of(i,tank_ci,zi+1))+")\n")

        #baffles
        outfile.write("\n\t\t//baffles\n")
        for i in range(1,nsplits,2): #all odd numbers
            outfile.write("\t\t( ")
            outfile.write(str(get_baffle_point_of(i,tank_ci,zi))+" ")
            outfile.write(str(get_baffle_point_of(i,tank_ci-1,zi))+" ")
            outfile.write(str(get_baffle_point_of(i,tank_ci-1,zi+1))+" ")
            outfile.write(str(get_baffle_point_of(i,tank_ci,zi+1))+")\n")

            outfile.write("\t\t( ")
            outfile.write(str(get_globalindex_of(i,tank_ci,zi))+" ")
            outfile.write(str(get_globalindex_of(i,tank_ci-1,zi))+" ")
            outfile.write(str(get_globalindex_of(i,tank_ci-1,zi+1))+" ")
            outfile.write(str(get_globalindex_of(i,tank_ci,zi+1))+")\n")

    #inlet wall patch 
    #skip rotor to hub to inhub_circ to polygon region 
    #which is covered in inflow
    zi=0
    outfile.write("\n\t\t//circles\n")
    for ci in range(rot_ci+1,ncirc): #start from rotor circle
        outfile.write("\n\t\t//circle "+str(ci)+" - "+str(ci-1)+" \n")
        for i in range(nsplits):
            outfile.write("\t\t( ")
            outfile.write(str(get_globalindex_of(i,ci,zi))+" ")
            outfile.write(str(get_globalindex_of(i+1,ci,zi))+" ")
            outfile.write(str(get_globalindex_of(i+1,ci-1,zi))+" ")
            outfile.write(str(get_globalindex_of(i,ci-1,zi))+")\n")

    outfile.write("\t)\n")

    #stitch faces patches
    outfile.write("\n\tempty inside_to_hub\n\t(\n")
    
    zi_pairs=[]
    for vols in nonbaff_volumes:
        zi_pairs.append([vols,vols+1])


    for zi_pair in zi_pairs:

        zi_below=zi_pair[0]
        zi_above=zi_pair[1]
        outfile.write("\n\t\t//pair :"+str(zi_below)+"-"+str(zi_above)+"\n")

        for i in range(0,nsplits,2): #even numbers
            outfile.write("\t\t( ")
            outfile.write(str(get_baffle_point_of(i,hub_ci,zi_below))+" ")
            outfile.write(str(get_baffle_point_of(i,inhub_ci,zi_below))+" ")
            outfile.write(str(get_baffle_point_of(i,inhub_ci,zi_above))+" ")
            outfile.write(str(get_baffle_point_of(i,hub_ci,zi_above))+")\n")

    outfile.write("\t)\n")
    
    outfile.write("\n\tempty inside_to_hub_copy\n\t(\n")

    for zi_pair in zi_pairs:

        zi_below=zi_pair[0]
        zi_above=zi_pair[1]
        outfile.write("\n\t\t//pair :"+str(zi_below)+"-"+str(zi_above)+"\n")

        for i in range(0,nsplits,2): #even numbers
            outfile.write("\t\t( ")
            outfile.write(str(get_globalindex_of(i,hub_ci,zi_below))+" ")
            outfile.write(str(get_globalindex_of(i,inhub_ci,zi_below))+" ")
            outfile.write(str(get_globalindex_of(i,inhub_ci,zi_above))+" ")
            outfile.write(str(get_globalindex_of(i,hub_ci,zi_above))+")\n")

    
    outfile.write("\t)\n")

    #stitch faces patches
    outfile.write("\n\tempty hub_to_rotor\n\t(\n")
    
    for zi_pair in zi_pairs:

        zi_below=zi_pair[0]
        zi_above=zi_pair[1]
        outfile.write("\n\t\t//pair :"+str(zi_below)+"-"+str(zi_above)+"\n")

        for i in range(0,nsplits,2): #even numbers
            outfile.write("\t\t( ")
            outfile.write(str(get_baffle_point_of(i,rot_ci,zi_below))+" ")
            outfile.write(str(get_baffle_point_of(i,hub_ci,zi_below))+" ")
            outfile.write(str(get_baffle_point_of(i,hub_ci,zi_above))+" ")
            outfile.write(str(get_baffle_point_of(i,rot_ci,zi_above))+")\n")

    outfile.write("\t)\n")
    outfile.write("\n\tempty hub_to_rotor_copy\n\t(\n")

    for zi_pair in zi_pairs:

        zi_below=zi_pair[0]
        zi_above=zi_pair[1]
        outfile.write("\n\t\t//pair :"+str(zi_below)+"-"+str(zi_above)+"\n")

        for i in range(0,nsplits,2): #even numbers
            outfile.write("\t\t( ")
            outfile.write(str(get_globalindex_of(i,rot_ci,zi_below))+" ")
            outfile.write(str(get_globalindex_of(i,hub_ci,zi_below))+" ")
            outfile.write(str(get_globalindex_of(i,hub_ci,zi_above))+" ")
            outfile.write(str(get_globalindex_of(i,rot_ci,zi_above))+")\n")

    
    outfile.write("\t)\n")

    outfile.write(");\n")


#main
outfile=open("blockMeshDict_reactor","w")
write_ofoam_preamble(outfile)
write_vertices(outfile)
write_edges(outfile)
write_blocks(outfile)
write_patches(outfile)
outfile.close()
