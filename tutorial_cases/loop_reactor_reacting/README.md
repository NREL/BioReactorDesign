### Loop reactor with chemical reaction

* Loop reactor scaled up with 608m3 of liquid
* Reaction kinetics transforming CO2 to CH4
* No dynamic mixing enabled (power = 0W in the fvModels) 
* Setup with constant diameter but can use PBE by changing the line 
`cp constant/phaseProperties_constantd constant/phaseProperties`
into 
`cp constant/phaseProperties_pbe constant/phaseProperties`
If you use pbe, you may change `system/controlDict` to print the bubble diameter
* writeGlobalVars.py writes `constant/globalVars` from `constant/globalVars_tmp`

Single core exec

1. `bash run.sh`


 
