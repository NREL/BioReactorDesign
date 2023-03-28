## Generate STL of spider sparger

### Execute without plotting

`python main.py -cr 0.25 -na 12 -aw 0.1 -al 0.5`

### Execute with plotting

`python main.py -v -cr 0.25 -na 12 -aw 0.1 -al 0.5`

Generates

<p float="left">
  <img src="image/simpleOutput.png" width="250"/>
</p>


### Manual

```
usage: main.py [-h] [-cr] [-na] [-aw] [-al] [-v]

Generate Spider Sparger STL

optional arguments:
  -h, --help            show this help message and exit
  -cr , --centerRadius 
                        Radius of the center distributor
  -na , --nArms         Number of spider arms
  -aw , --armsWidth     Width of spider arms
  -al , --armsLength    Length of spider arms
  -v, --verbose         plot on screen

```

## Generate fi.gas

`cd inhomogeneousBC`

### Execute

Generates `fi.gas` in `IC_inhomo`. If `r<0.1` use pores of diameter `3e-5`. Gradually decrease the pore size to `2e-5` linearly.

`python main.py -rc 0.1 -re 1 -pi 3e-5 -po 2e-5 -xc 0 -zc 0`

### Execute with logging

`python main.py -rc 0.1 -re 1 -pi 3e-5 -po 2e-5 -xc 0 -zc 0 -v`

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





