#!/bin/bash
#SBATCH --qos=high
#SBATCH --job-name=OFpost
##SBATCH --partition=standard
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=36
#SBATCH --time=4:00:00
#SBATCH --account=ifsheat
#SBATCH --output=log_post.out
###SBATCH --dependency=afterany:8439863

source /projects/gas2fuels/ofoam_cray_mpich/OpenFOAM-dev/etc/bashrc

# post-process
#reconstructPar -newTimes
reconstructPar -time 1.5,1.6,1.7,1.8,1.9,2,2.1,2.2,2.3,2.4

touch soln.foam
module purge
ml paraview
pvpython get_avg_conc.py
pvpython find_sup_velocity.py
pvpython massflow_outlet.py
