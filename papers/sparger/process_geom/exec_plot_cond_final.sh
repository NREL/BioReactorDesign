rootFolder=/Users/mhassana/Desktop/GitHub/BioReactorDesign_mar4/papers/sparger
fields="CO.gas CO.liquid CO2.gas CO2.liquid H2.gas H2.liquid alpha.gas d.gas kla_H2 kla_CO kla_CO2"

plotCond () {
    python plot_cond.py -sf $1 -cp $2 -fl $fields -pl $3 -ff $4 -p $5
}
plotCond_exc () {
    python plot_cond.py -sf $1 -cp $2 -fl $fields -pl $3 -ff $4 -p $5 -cfe $6
}

plotCond_exc $rootFolder/study_coarse_flatDonut flat_donut width flat_donut_coarse param_flatDonut.npz "flat_donut_0 flat_donut_1 flat_donut_2 flat_donut_3 flat_donut_4"
#plotCond_exc $rootFolder/study_coarse_multiRing multiRing width multiRing_coarse param_multiRing.npz "multiRing_0 multiRing_1 multiRing_2 multiRing_3" 
#plotCond_exc $rootFolder/study_coarse_sideSparger side_sparger height side_sparger_coarse param_sideSparger.npz "side_sparger_0 side_sparger_1 side_sparger_2"

#plotCond_exc $rootFolder/study_fine_flatDonut flat_donut width flat_donut_fine param_flatDonut.npz "flat_donut_0 flat_donut_1 flat_donut_2 flat_donut_3 flat_donut_4"



