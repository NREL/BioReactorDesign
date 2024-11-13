* Scaled up loop reactor with 608m3 of liq
* Reaction kinetics transforming CO2 to CH4
* No mixers enabled (power = 0 in the fvModels) 
* Setup with constant diameter but can use PBE by changing the line 
`cp constant/phaseProperties_constantd constant/phaseProperties`
into 
`cp constant/phaseProperties_pbe constant/phaseProperties`
* writeGlobalVars.py writes `constant/globalVars` from `constant/globalVars_tmp`


## Running the case

bash run.sh

 
