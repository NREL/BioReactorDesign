fields="GH GH_height d CO2_liq CO_liq H2_liq kla_CO2 kla_CO kla_H2"
diff_name="D_H2 D_CO2 D_CO"
diff_val="1.2097e-8 1.4663e-9 3.9518e-9"
#conda activate /projects/gas2fuels/postProc_env

computeQOI () {
    for dir in $1/*/; do
        python compute_QOI.py -f $dir -avg 5 -conv 100 -vl $fields
    done
}

rootFolder=/projects/disrupt/mhassana/sparger_geom_coarse_june21/
computeQOI $rootFolder/study_coarse_flatDonut -dn $diff_name -dv $diff_val
computeQOI $rootFolder/study_coarse_multiRing -dn $diff_name -dv $diff_val
computeQOI $rootFolder/study_coarse_sideSparger -dn $diff_name -dv $diff_val

rootFolder=/projects/disrupt/mhassana/sparger_geom_fine_june21/
computeQOI $rootFolder/study_fine_flatDonut -dn $diff_name -dv $diff_val
computeQOI $rootFolder/study_fine_multiRing -dn $diff_name -dv $diff_val
computeQOI $rootFolder/study_fine_multiRing_num -dn $diff_name -dv $diff_val
computeQOI $rootFolder/study_fine_sideSparger -dn $diff_name -dv $diff_val
