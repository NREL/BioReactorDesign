import sys
import os
import math
import simParameters
def fprintf(stream, format_spec, *args):
    stream.write(format_spec % args)


# CREATE AND RUN CASES
def createCase(gas_species,vol,VVM,grav,vFac):

    baseDir = os.getcwd()
    # Create case name
    caseName = "cases/" + gas_species +"_vol_"+ str(vol)+"_VVM_"+str(VVM)+"_g_"+str(grav)+"_vfac_"+str(vFac)
    print(caseName)
    os.system("rm -rf " + caseName)
    os.system("mkdir -p " + caseName)
    os.system("cp -r base_case/* " + caseName + "/.")
    
    volRatio = ((vol/0.005)**(1.0/3.0)) #This is based on the 5L case

    inlet_area = ((vol/0.0005)**(2.0/3.0))*0.000253597 #This is based on the 0.5L case

    Uin = VVM*vol/(60*inlet_area*0.5)
    
    # Open file and return error if file exixts already
    f = open(caseName + "/constant/caseVars", "w")
    fprintf(f,"VVM %.10f;\n",VVM)
    fprintf(f,"liqVol %.10f;\n",vol)
    fprintf(f,"HtBcol %.10f;\n",volRatio*0.291)
    fprintf(f,"DiaBcol %.10f;\n",volRatio*0.181)
    fprintf(f,"LiqHt %.10f;\n",volRatio*0.1847)
    fprintf(f,"grav %.10f;\n",grav)
    fprintf(f,"Uin %.10f;\n",Uin)
    fprintf(f,"viscousFactor %.10f;\n",vFac)
    
    if gas_species == "air":
        fprintf(f,"N2_in %.10f;\n",0.79)
        fprintf(f,"O2_in %.10f;\n",0.21)
        fprintf(f,"H2_in %.10f;\n",0)
        fprintf(f,"CO2_in %.10f;\n",0)     
        fprintf(f,"sigmaLiq %.10f;\n",0.072)
    elif gas_species == "CO2H2":   
        fprintf(f,"N2_in %.10f;\n",0)
        fprintf(f,"O2_in %.10f;\n",0)
        fprintf(f,"H2_in %.10f;\n",0.155)
        fprintf(f,"CO2_in %.10f;\n",0.845)     
        fprintf(f,"sigmaLiq %.10f;\n",0.072)
    elif gas_species == "CO2": 
        fprintf(f,"N2_in %.10f;\n",0)
        fprintf(f,"O2_in %.10f;\n",0)
        fprintf(f,"H2_in %.10f;\n",0)
        fprintf(f,"CO2_in %.10f;\n",1)     
        fprintf(f,"sigmaLiq %.10f;\n",0.072)
    elif gas_species == "O2": 
        fprintf(f,"N2_in %.10f;\n",0)
        fprintf(f,"O2_in %.10f;\n",1)
        fprintf(f,"H2_in %.10f;\n",0)
        fprintf(f,"CO2_in %.10f;\n",0)     
        fprintf(f,"sigmaLiq %.10f;\n",0.072)  
    f.close()


    f = open(caseName + "/runjob", "w")
    fprintf(f, "#!/bin/bash\n")
    fprintf(f, "##SBATCH --qos=high\n")
    fprintf(f, "#SBATCH --job-name=%s\n",caseName)
    #fprintf(f, "#SBATCH --partition=debug\n")
    fprintf(f, "#SBATCH --nodes=1\n")
    fprintf(f, "#SBATCH --ntasks-per-node=30  # use 52\n")
    fprintf(f, "#SBATCH --time=2-00:00:00\n")
   # fprintf(f, "#SBATCH --time=0-01:00:00\n")
    fprintf(f, "#SBATCH --account=biospace\n")
    fprintf(f, "#SBATCH --error=log.err\n")
    fprintf(f, "\n")
    fprintf(f, "module purge\n")
    fprintf(f, "ml PrgEnv-cray\n")
    fprintf(f, "ml openfoam/13-craympich-scotch\n")
    fprintf(f, "decomposePar -latestTime -fileHandler collated\n")
    fprintf(f, "srun -n 30 --cpu_bind=cores foamRun -parallel -fileHandler collated\n")
    fprintf(f, "\n")
    fprintf(f, "reconstructPar -newTimes\n")
    f.close()

    os.chdir(baseDir+"/"+caseName)
    if vol == 0.0005:
        os.system("mv constant/polyMesh_05 constant/polyMesh")
    else:
        os.system("mv constant/polyMesh_50 constant/polyMesh")

    os.system("sbatch runjob")    

    os.chdir(baseDir)

# CREATE AND RUN CASES
for gas_species in simParameters.gas_species:
    for vol in simParameters.vol:
        for VVM in simParameters.VVM:
            for grav in simParameters.grav:
                for vFac in simParameters.viscosityFactor:
                    createCase(gas_species,vol,VVM,grav,vFac)
