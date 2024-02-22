#!/bin/bash
##SBATCH --qos=high
#SBATCH --job-name=bubbleCol
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36
#SBATCH --time=2-00:00:00
#SBATCH --account=gas2fuels
#SBATCH --output=log.out
##SBATCH --dependency=afterany:8439863

module purge
source /projects/gas2fuels/load_OF9_pbe
TMPDIR=/tmp/scratch/

#./Allrun
decomposePar -fileHandler collated -latestTime
srun -n 36 multiphaseEulerFoam -parallel -fileHandler collated

# post-process
reconstructPar -newTimes

touch sol.foam
module purge
ml paraview/5.8.1-gui
pvpython get_avg_conc.py

pvpython find_sup_velocity.py
