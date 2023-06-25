fields="GH GH_height d CO2_liq CO_liq H2_liq kla_CO2 kla_CO kla_H2"
rootFolder=../sparger_geom/study_coarse_flatDonut/

python compute_QoI.py -f $rootFolder/flat_donut_0/  -avg 2 -conv 10 -vl $fields
python compute_QoI.py -f $rootFolder/flat_donut_1/  -avg 2 -conv 10 -vl $fields
python compute_QoI.py -f $rootFolder/flat_donut_2/  -avg 2 -conv 10 -vl $fields
python compute_QoI.py -f $rootFolder/flat_donut_3/  -avg 2 -conv 10 -vl $fields
