#!/bin/bash
##SBATCH --qos=high
#SBATCH --job-name=pilot
##SBATCH --partition=standard
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=50  # use 52
#SBATCH --time=2-00:00:00
#SBATCH --account=hdcomb
#SBATCH --output=log.out
#SBATCH --error=log.err
###SBATCH --dependency=afterany:8439863
#SBATCH --distribution=cyclic:cyclic

module purge
ml PrgEnv-cray
source /projects/gas2fuels/ofoam_cray_mpich/OpenFOAM-dev/etc/bashrc

#. ./presteps.sh
#decomposePar -latestTime -fileHandler collated 
srun -n 200 --cpu_bind=cores multiphaseEulerFoam -parallel -fileHandler collated

# post-process
#reconstructPar -newTimes

