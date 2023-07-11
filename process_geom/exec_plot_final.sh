fields="GH GH_height d CO2_liq CO_liq H2_liq kla_CO2 kla_CO kla_H2"

plotQOI () {
    python plot_qoi.py -sf $1 -cp $2 -vl $fields -pl $3 -ff $4 -p $5
}
plotQOI_exc () {
    python plot_qoi.py -sf $1 -cp $2 -vl $fields -pl $3 -ff $4 -p $5 -cfe $6
}


rootFolder=/Users/mhassana/Desktop/GitHub/spargerDesign_june25/caseResults_geom/coarse
#plotQOI_exc $rootFolder/study_coarse_flatDonut flat_donut width flat_donut_coarse param_flatDonut.npz "flat_donut_0 flat_donut_1 flat_donut_2 flat_donut_3 flat_donut_4"
plotQOI $rootFolder/study_coarse_multiRing multiRing "width spacing" multiRing_coarse param_multiRing.npz ""
plotQOI_exc $rootFolder/study_coarse_sideSparger side_sparger height side_sparger_coarse param_sideSparger.npz "side_sparger_0"

rootFolder=/Users/mhassana/Desktop/GitHub/spargerDesign_june25/caseResults_geom/fine
#plotQOI_exc $rootFolder/study_fine_flatDonut flat_donut width flat_donut_fine param_flatDonut.npz "flat_donut_0 flat_donut_1 flat_donut_2 flat_donut_3 flat_donut_4"


#plotQOI $rootFolder/study_fine_multiRing multiRing width multiRing_fine param_multiRing.npz
#plotQOI $rootFolder/study_fine_multiRing multiRing spacing multiRing_fine param_multiRing.npz
#plotQOI $rootFolder/study_fine_sideSparger side_sparger height side_sparger_fine param_sideSparger.npz
