rootFolder=/Users/mhassana/Desktop/GitHub/BioReactorDesign_mar4/papers/sparger

fields="GH GH_height d CO2_liq CO_liq H2_liq kla_CO2 kla_CO kla_H2"

plotQOI () {
    python plot_qoi.py -sf $1 -cp $2 -vl $fields -pl $3 -ff $4 -p $5
}
plotQOI_exc () {
    python plot_qoi.py -sf $1 -cp $2 -vl $fields -pl $3 -ff $4 -p $5 -cfe $6
}
compQOI () {
    python compare_qoi.py -df $1 -m $2 -xl $3  -yl $4 -ff $5 -n $6 -l $7
}
plotQOI_exc_bound () {
    python plot_qoi.py -sf $1 -cp $2 -vl CO2_liq CO_liq H2_liq -pl $3 -ff $4 -p $5 -cfe $6 -vmin $7 -vmax $8
}


plotQOI_exc $rootFolder/study_coarse_flatDonut flat_donut width flat_donut_coarse param_flatDonut.npz "flat_donut_0 flat_donut_1 flat_donut_2 flat_donut_3 flat_donut_4"
plotQOI_exc $rootFolder/study_coarse_multiRing multiRing width multiRing_coarse param_multiRing.npz "multiRing_0 multiRing_1 multiRing_2 multiRing_3"
plotQOI_exc $rootFolder/study_coarse_sideSparger side_sparger height side_sparger_coarse param_sideSparger.npz "side_sparger_0 side_sparger_1 side_sparger_2"

plotQOI_exc $rootFolder/study_fine_flatDonut flat_donut width flat_donut_fine param_flatDonut.npz "flat_donut_0 flat_donut_1 flat_donut_2 flat_donut_3 flat_donut_4"
plotQOI_exc $rootFolder/study_fine_multiRing multiRing width multiRing_fine param_multiRing.npz "multiRing_0 multiRing_1 multiRing_2 multiRing_3"
plotQOI_exc $rootFolder/study_fine_sideSparger side_sparger height side_sparger_fine param_sideSparger.npz "side_sparger_0 side_sparger_1 side_sparger_2"

df_coarse="Data/flat_donut_coarse/qoi/width/"
df_fine="Data/flat_donut_fine/qoi/width/"
compQOI "$df_coarse/CO2_liq.npz $df_fine/CO2_liq.npz" 1d width CO2_liq flat_donut_mesh CO2_liq "coarse fine" 
compQOI "$df_coarse/CO_liq.npz $df_fine/CO_liq.npz" 1d width CO_liq flat_donut_mesh CO_liq "coarse fine" 
compQOI "$df_coarse/H2_liq.npz $df_fine/H2_liq.npz" 1d width H2_liq flat_donut_mesh H2_liq "coarse fine" 


