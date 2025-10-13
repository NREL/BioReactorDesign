#!/bin/bash
#SBATCH --job-name=Biomethan
#SBATCH --partition=debug
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=36
#SBATCH --time=1:00:00
#SBATCH --account=gas2fuels
#SBATCH --output=log.out

source /projects/gas2fuels/ofoam_cray_mpich/OpenFOAM-dev/etc/bashrc

./presteps.sh
setFields
decomposePar -latestTime -fileHandler collated 
srun -n 32 multiphaseEulerFoam -parallel -fileHandler collated

# post-process
reconstructPar -newTimes

