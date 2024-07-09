#!/bin/bash
#SBATCH --qos=high
#SBATCH --job-name=test_laak
##SBATCH --partition=debug
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=50
#SBATCH --time=48:00:00
#SBATCH --account=bpms
#SBATCH --output=log.out

source /projects/gas2fuels/ofoam_cray_mpich/OpenFOAM-dev/etc/bashrc

bash presteps.sh
decomposePar -fileHandler collated
srun -n 200 multiphaseEulerFoam -parallel -fileHandler collated
