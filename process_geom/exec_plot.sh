fields="GH GH_height d CO2_liq CO_liq H2_liq kla_CO2 kla_CO kla_H2"
rootFolder=../../spargerDesign_june16/sparger_geom/study_coarse_flatDonut/

python plot_qoi.py -sf $rootFolder -cp flat_donut -vl $fields -pl width -ff flat_donut -p param_flatDonut.npz -cfe flat_donut_4 
