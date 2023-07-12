root=/lustre/eaglefs/projects/gas2fuels/ValidationExp2_final




python3 compute_observations.py -avg 5 -f $root/binBreakupLuo_17_mesh1
python3 compute_observations.py -avg 5 -f $root/binBreakupLuo_19_mesh1
python3 compute_observations.py -avg 5 -f $root/breakup_17_mesh1_normDist
python3 compute_observations.py -avg 5 -f $root/breakup_19_mesh1_normDist
python3 compute_observations.py -avg 5 -f $root/binBreakup_17_mesh1_normDist
python3 compute_observations.py -avg 5 -f $root/binBreakup_19_mesh1_normDist



#python3 compute_observations.py -avg 5 -f $root/constD_17_mesh1
#python3 compute_observations.py -avg 5 -f $root/constD_17_mesh2
#python3 compute_observations.py -avg 5 -f $root/constD_19_mesh1
#python3 compute_observations.py -avg 5 -f $root/constD_19_mesh2
#python3 compute_observations.py -avg 5 -f $root/breakup_17_mesh1
#python3 compute_observations.py -avg 5 -f $root/breakup_17_mesh2
#python3 compute_observations.py -avg 5 -f $root/breakup_17_f2
#python3 compute_observations.py -avg 5 -f $root/breakup_19_mesh1
#python3 compute_observations.py -avg 5 -f $root/breakup_19_mesh2
#python3 compute_observations.py -avg 5 -f $root/breakup_19_f2
#python3 compute_observations.py -avg 5 -f $root/binBreakup_17_mesh1
#python3 compute_observations.py -avg 5 -f $root/binBreakup_17_mesh2
#python3 compute_observations.py -avg 5 -f $root/binBreakup_17_f2
#python3 compute_observations.py -avg 5 -f $root/binBreakup_19_mesh1
#python3 compute_observations.py -avg 5 -f $root/binBreakup_19_mesh2
#python3 compute_observations.py -avg 5 -f $root/binBreakup_19_f2

