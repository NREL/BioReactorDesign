fields="GH GH_height d CO2_liq CO_liq H2_liq kla_CO2 kla_CO kla_H2"
rootFolder=/Users/mhassana/Desktop/GitHub/spargerDesign_june25/caseResults
python plot_qoi_multiFold.py -sf $rootFolder/pore_size_2mm $rootFolder/widerBCR -pv 0.72 2.16 -pn columnDiameter -ff columnDiameter
