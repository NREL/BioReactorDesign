## Generate fi.gas

### Execute

Generates `fi.gas` in `IC_inhomo`. If `r<0.1` use pores of diameter `3e-5`. Gradually decrease the pore size to `2e-5` linearly.

`python main.py -rc 0.1 -re 1 -pi 3e-5 -po 2e-5 -xc 0 -zc 0 -ugs 0.01 -ds 0.15`

### Execute with logging

`python main.py -rc 0.1 -re 1 -pi 3e-5 -po 2e-5 -xc 0 -zc 0 -ugs 0.01 -ds 0.15 -v`

### Manual

```
usage: main.py [-h] [-v] [-rc] [-re] [-pi] [-po] [-xc] [-zc] [-ds] [-ugs]

Generate inhomogeneous boundary

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         plot on screen
  -rc , --r_const       Constant radius value
  -re , --r_end         End radius value
  -pi , --pore_in       Pore diameter at center
  -po , --pore_out      Pore diameter at radius end
  -xc , --xcent         Column center x
  -zc , --zcent         Column center z
  -ds , --diam_sparger 
                        Sparger diameter
  -ugs , --superf_vel   Superficial velocity

```
