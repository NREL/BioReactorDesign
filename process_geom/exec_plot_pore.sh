fields="GH GH_height d CO2_liq CO_liq H2_liq kla_CO2 kla_CO kla_H2"
fields_cond="CO.gas CO.liquid CO2.gas CO2.liquid H2.gas H2.liquid alpha.gas d.gas kla_CO2 kla_CO kla_H2"
rootFolder=/Users/mhassana/Desktop/GitHub/spargerDesign_june25/caseResults

sFold=' '\
"$rootFolder/pore_size_1mm "\
"$rootFolder/pore_size_5mm "

pVal=' '\
"0.001 "\
"0.005 "


python plot_qoi_multiFold.py -sf $sFold -pv $pVal -pn bubbleDiam -ff bubbleDiameter -vl $fields

python plot_cond_multiFold.py -sf $sFold -pv $pVal -pn bubbleDiam -ff bubbleDiameter -fl $fields_cond
