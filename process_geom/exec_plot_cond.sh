fields="CO.gas CO.liquid CO2.gas CO2.liquid H2.gas H2.liquid alpha.gas d.gas kla"
rootFolder=../sparger_geom/study_coarse_flatDonut/

python plot_conditionalMean_geom.py -sf $rootFolder -cp flat_donut -fl $fields -pl width -ff flat_donut -p param_flatDonut.npz -cfe flat_donut_4

