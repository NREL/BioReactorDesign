fields="alpha.gas CO.gas CO.liquid CO2.gas CO2.liquid H2.gas H2.liquid d.gas kla"

computeCond () {
    for dir in $1/*/; do
        python compute_conditionalMean.py -f $dir -avg 5 -fl $fields
    done
}

rootFolder=/projects/disrupt/mhassana/sparger_geom_coarse_june21/
computeCond $rootFolder/study_coarse_flatDonut
computeCond $rootFolder/study_coarse_multiRing
computeCond $rootFolder/study_coarse_sideSparger


rootFolder=/projects/disrupt/mhassana/sparger_geom_fine_june21/
computeCond $rootFolder/study_fine_flatDonut
computeCond $rootFolder/study_fine_multiRing
computeCond $rootFolder/study_fine_multiRing_num
computeCond $rootFolder/study_fine_sideSparger
