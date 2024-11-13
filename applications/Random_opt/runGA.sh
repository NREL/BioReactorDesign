for ga_iter in NiterGA:
    1) decide next batch of GA samples
    2) save samples as 'samples_batch${ga_iter}.npy'
    3) setup the next batch of CFD runs
    python ga2sim.py -bf GAbatch${ga_iter}_0.4vvm_6kW  -vvm 0.4 -pow 6000 --ga_sample_file samples_batch${ga_iter}.npy
    4) run the CFD sim
    cd GAbatch${ga_iter}_0.4vvm_6kW
    bash many_scripts_start
    5) waituntil done
    6) compute QOI
    cd GAbatch${ga_iter}_0.4vvm_6kW
    bash many_scripts_post
    7) Store qoi of batch in numpy array
    python read_qoi.py -bf GAbatch${ga_iter}_0.4vvm_6kW 
