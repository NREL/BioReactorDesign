if test -f sigma.log; then
    rm sigma.log
fi

python calibration_uq_sigm.py -qoi gh -id 17
python calibration_uq_sigm.py -qoi gh -id 19
python calibration_uq_sigm.py -qoi co2 -id 17
python calibration_uq_sigm.py -qoi co2 -id 19
python calibration_uq_multi.py
