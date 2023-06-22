fields="alpha.gas CO.gas CO.liquid CO2.gas CO2.liquid H2.gas H2.liquid d.gas"
rootFolder=../sparger_geom/study_coarse_flatDonut/

python compute_conditionalMean.py -f $rootFolder/flat_donut_0/  -avg 2 -fl $fields
python compute_conditionalMean.py -f $rootFolder/flat_donut_1/  -avg 2 -fl $fields
python compute_conditionalMean.py -f $rootFolder/flat_donut_2/  -avg 2 -fl $fields
python compute_conditionalMean.py -f $rootFolder/flat_donut_3/  -avg 2 -fl $fields

