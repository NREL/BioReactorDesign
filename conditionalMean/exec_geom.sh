fields="alpha.gas CO.gas CO.liquid CO2.gas CO2.liquid H2.gas H2.liquid d.gas kla_H2 kla_CO kla_CO2"
diff_name="D_H2 D_CO2 D_CO"
diff_val="1.2097e-8 1.4663e-9 3.9518e-9"
rootFolder=../../spargerDesign_june16/sparger_geom/study_coarse_flatDonut/

python compute_conditionalMean.py -f $rootFolder/flat_donut_0/  -avg 2 -fl $fields -dn $diff_name -dv $diff_val
python compute_conditionalMean.py -f $rootFolder/flat_donut_1/  -avg 2 -fl $fields -dn $diff_name -dv $diff_val
python compute_conditionalMean.py -f $rootFolder/flat_donut_2/  -avg 2 -fl $fields -dn $diff_name -dv $diff_val
python compute_conditionalMean.py -f $rootFolder/flat_donut_3/  -avg 2 -fl $fields -dn $diff_name -dv $diff_val

