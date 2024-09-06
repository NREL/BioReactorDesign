if [ ! -f qoi.txt ]; then
    # Reconstruct if needed
    source /projects/gas2fuels/ofoam_cray_mpich/OpenFOAM-dev/etc/bashrc
    reconstructPar -newTimes
    module load anaconda3/2022.05
    conda activate /projects/gas2fuels/conda_env/spargerDesign
    python read_history.py -cr .. -cn local -df data
    python get_qoi.py
    conda deactivate
else
   echo "WARNING: QOI already computed"
fi

