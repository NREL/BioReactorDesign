fields="CO.gas CO.liquid CO2.gas CO2.liquid H2.gas H2.liquid alpha.gas d.gas kla_CO kla_CO2 kla_H2"
rootFolder=../../spargerDesign_june16/sparger_geom/study_coarse_flatDonut/

python plot_cond.py -sf $rootFolder -cp flat_donut -fl $fields -pl width -ff flat_donut -p param_flatDonut.npz -cfe flat_donut_4

