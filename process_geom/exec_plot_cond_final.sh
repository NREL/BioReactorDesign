fields="CO.gas CO.liquid CO2.gas CO2.liquid H2.gas H2.liquid alpha.gas d.gas kla_H2 kla_CO kla_CO2"

plotCond () {
    python plot_conditionalMean_geom.py -sf $1 -cp $2 -fl $fields -pl $3 -ff $4 -p $5
}

rootFolder=/Users/mhassana/Desktop/GitHub/spargerDesign_june25/caseResults_geom/coarse
plotCond $rootFolder/study_coarse_flatDonut flat_donut width flat_donut_coarse param_flatDonut.npz
plotCond $rootFolder/study_coarse_multiRing multiRing width multiRing_coarse param_multiRing.npz
plotCond $rootFolder/study_coarse_multiRing multiRing spacing multiRing_coarse param_multiRing.npz
plotQOI $rootFolder/study_coarse_sideSparger side_sparger height side_sparger_coarse param_sideSparger.npz

rootFolder=/Users/mhassana/Desktop/GitHub/spargerDesign_june25/caseResults_geom/fine
plotCond $rootFolder/study_fine_flatDonut flat_donut width flat_donut_fine param_flatDonut.npz
#plotCond $rootFolder/study_fine_multiRing multiRing width multiRing_fine param_multiRing.npz
#plotCond $rootFolder/study_fine_multiRing multiRing spacing multiRing_fine param_multiRing.npz
#plotCond $rootFolder/study_fine_sideSparger side_sparger height side_sparger_fine param_sideSparger.npz

