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




## Generate 3D sparger

`cd blockCyclindricalMeshing`

### Execute

Generates `blockMeshDict` in `system`

```
root=`pwd`
caseFolder=case

python writeBlockMesh.py -i sideSparger/input.json -g sideSparger/geometry.json -o $caseFolder/system

cd $caseFolder
blockMesh
transformPoints "scale=(0.001 0.001 0.001)"
transformPoints "rotate=((0 0 1) (0 1 0))"
cd $root
```

Will generate this

<p float="left">
  <img src="image/3dsparger.png" width="250"/>
</p>


### How to change the dimensions or mesh refinement

All dimensions and mesh are controlled by the input file `input.json`. 


### How to change the arrangement of concentric cylinders

The block topology is controlled by the `geometry.json`
Always work with a schematic. Here is the schematic for this case

<p float="left">
  <img src="image/schematic.png" width="250"/>
</p>

The purple blocks are walls (not meshed) and the white blocks are fluid blocks (meshed). The symmetry axis is indicated as a dashed line

In the `geometry.json`, the purple blocks are defined as

```
"Walls": {
                "Support": [
                            {"R": 0, "L": 3},
                            {"R": 1, "L": 3}
                           ],
                "Sparger": [
                            {"R": 0, "L": 2},
                            {"R": 1, "L": 2},
                            {"R": 2, "L": 2}
                           ]
        }
```

### How to change boundaries

Boundaries are defined with three types, `top`, `bottom` and `lateral`

In the case of sparger walls shown below with the red lines
<p float="left">
  <img src="image/schematicSpargerWalls.png" width="250"/>
</p>

the boundary is defined in the `geometry.json` as
```
"Boundary": {
                "wall_sparger":[
                           {"type": "bottom", "Rmin": 2, "Rmax": 2, "Lmin": 2, "Lmax": 3},
                           {"type": "top", "Rmin": 0, "Rmax": 0, "Lmin": 1, "Lmax": 2},
                           {"type": "top", "Rmin": 1, "Rmax": 1, "Lmin": 1, "Lmax": 2},
                           {"type": "top", "Rmin": 2, "Rmax": 2, "Lmin": 1, "Lmax": 2}
                         ],
...
```

In the case of sparger inlet shown below with the red line
<p float="left">
  <img src="image/schematicSpargerInlet.png" width="250"/>
</p>

the boundary is defined in the `geometry.json` as
```
"Boundary": {
                "inlet": [
                           {"type": "lateral", "Rmin": 2, "Rmax": 3, "Lmin": 2, "Lmax": 2}
                         ],
...
```

### Acknowledgments

This work was authored by the National Renewable Energy Laboratory (NREL), operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308. This work was supported by funding from DOE's Bioenergy Technologies Office (BETO) program. The research was performed using computational resources sponsored by the Department of Energy's Office of Energy Efficiency and Renewable Energy and located at the National Renewable Energy Laboratory. The views expressed in the article do not necessarily represent the views of the DOE or the U.S. Government. The U.S. Government retains and the publisher, by accepting the article for publication, acknowledges that the U.S. Government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this work, or allow others to do so, for U.S. Government purposes.



