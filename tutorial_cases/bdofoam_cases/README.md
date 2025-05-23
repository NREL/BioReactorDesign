### BDOFOAM run steps

1. Run the non-reacting case (bubble column here) to steady state, in this case until about 6000 sec
use the run_nonreact.sh script

2. The non-reacting steady state reconstructured file is the starting point for the react case. 
Use the run_react.sh script with any of your modifications to run the reacting case.


### Some other settings for react case

1. Change the startTime in system/controldict to the current reconstructed file.
Change final time based on how many fluid updates you will do in a 12 hour reaction 
period. 

2. In react/constant/microbeUpdateProperties:  
Set fluid update time - this is the fluid solver is run between 2 reaction updates.
set reaction update time - I am using 2 hours. But need to play with this to get an 
optimal update time that is computational efficient and yields convergent results when 
coupled with CFD.

3. The code outputs a text file called timehist.dat. This gives a table with
time, microbe_conc,  glucose, Xylose, Acetoin, BDO, OUR. There is also a file called wellmixed.dat that the 
code generates. This is the well-mixed case without CFD which is done to verify the 0d reaction model 
described in Sitaraman et al. "A reacting multiphase computational flow model for 
2, 3-butanediol synthesis in industrial-scale bioreactors." 
Chemical Engineering Research and Design 197 (2023): 38-52.


4. Use pv_extract_analyse script to reconstruct plotfiles and average O2 and hold-up.run pvjob.
