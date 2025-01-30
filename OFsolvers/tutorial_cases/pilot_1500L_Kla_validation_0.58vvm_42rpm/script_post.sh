#!/bin/bash
##SBATCH --qos=high
#SBATCH --job-name=Biomethan
##SBATCH --partition=standard
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=36
#SBATCH --time=4:00:00
#SBATCH --account=hdcomb
#SBATCH --output=log_post.out
###SBATCH --dependency=afterany:8439863

source /projects/gas2fuels/ofoam_cray_mpich/OpenFOAM-dev/etc/bashrc

#. ./presteps.sh
#decomposePar -latestTime -fileHandler collated 
#srun -n 320 multiphaseEulerFoam -parallel -fileHandler collated

# post-process
reconstructPar -newTimes

ml paraview/5.11.0-gui
pvpython get_avg_conc.py
#pvpython find_sup_velocity.py
pvpython get_kla.py
