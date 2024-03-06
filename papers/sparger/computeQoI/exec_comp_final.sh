fields="GH GH_height d CO2_liq CO_liq H2_liq kla_CO2 kla_CO kla_H2"
diff_name="D_H2 D_CO2 D_CO"
diff_val="1.2097e-8 1.4663e-9 3.9518e-9"
#conda activate /projects/gas2fuels/postProc_env

computeQOI () {
    for dir in $1/*/; do
        python compute_QOI.py -f $dir -avg 5 -conv 100 -vl $fields -dn $diff_name -dv $diff_val
    done
}
computeCond () {
    fields="alpha.gas CO.gas CO.liquid CO2.gas CO2.liquid H2.gas H2.liquid d.gas"
    for dir in $1/*/; do
        python compute_conditionalMean.py -f $dir -avg 5 -fl $fields
    done
}


rootFolder=/Users/mhassana/Desktop/GitHub/BioReactorDesign_mar4/papers/sparger
computeQOI $rootFolder/study_coarse_flatDonut
computeQOI $rootFolder/study_coarse_multiRing
computeQOI $rootFolder/study_coarse_sideSparger

computeCond $rootFolder/study_coarse_flatDonut
computeCond $rootFolder/study_coarse_multiRing
computeCond $rootFolder/study_coarse_sideSparger


rootFolder=/Users/mhassana/Desktop/GitHub/BioReactorDesign_mar4/papers/sparger

computeQOI $rootFolder/study_fine_flatDonut
computeQOI $rootFolder/study_fine_multiRing
computeQOI $rootFolder/study_fine_sideSparger
computeCond $rootFolder/study_fine_flatDonut
computeCond $rootFolder/study_fine_multiRing
computeCond $rootFolder/study_fine_sideSparger
