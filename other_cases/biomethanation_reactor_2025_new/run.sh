module purge
ml PrgEnv-cray
source /projects/gas2fuels/ofoam_cray_mpich/OpenFOAM-dev/etc/bashrc

. ./presteps.sh
decomposePar -latestTime -fileHandler collated 
srun -n 100 multiphaseEulerFoam -parallel -fileHandler collated
