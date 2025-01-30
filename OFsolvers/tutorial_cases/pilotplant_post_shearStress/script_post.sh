#!/bin/bash
#SBATCH --qos=high
#SBATCH --job-name=Biomethan
##SBATCH --partition=standard
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=36
#SBATCH --time=4:00:00
#SBATCH --account=microenviro
#SBATCH --output=log_post.out
###SBATCH --dependency=afterany:8439863

source /projects/gas2fuels/ofoam_cray_mpich/OpenFOAM-dev/etc/bashrc

#. ./presteps.sh
#decomposePar -latestTime -fileHandler collated 
#srun -n 320 multiphaseEulerFoam -parallel -fileHandler collated

# post-process
#reconstructPar -newTimes
multiphaseEulerFoam -postProcess -func "shearStress(phase=liquid)"
