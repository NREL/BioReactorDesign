## Generate 3D sparger

### Execute

Generates `blockMeshDict` in `system`

```
root=`pwd`
caseFolder=case

python writeBlockMesh.py -i sideSparger/input.json -t sideSparger/topology.json -o $caseFolder/system

cd $caseFolder
blockMesh
transformPoints "scale=(0.001 0.001 0.001)"
transformPoints "rotate=((0 0 1) (0 1 0))"
cd $root
```

Will generate this

<p float="left">
  <img src="../image/3dsparger.png" width="250"/>
</p>


### How to change the dimensions or mesh refinement

All dimensions and mesh are controlled by the input file `input.json`. 


### How to change the arrangement of concentric cylinders

The block topology is controlled by the `topology.json`
Always work with a schematic. Here is the schematic for this case

<p float="left">
  <img src="../image/schematic.png" width="250"/>
</p>

The purple blocks are walls (not meshed) and the white blocks are fluid blocks (meshed). The symmetry axis is indicated as a dashed line

In the `topology.json`, the purple blocks are defined as

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
  <img src="../image/schematicSpargerWalls.png" width="250"/>
</p>

the boundary is defined in the `topology.json` as
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
  <img src="../image/schematicSpargerInlet.png" width="250"/>
</p>

the boundary is defined in the `topology.json` as
```
"Boundary": {
                "inlet": [
                           {"type": "lateral", "Rmin": 2, "Rmax": 3, "Lmin": 2, "Lmax": 2}
                         ],
...
```

### Manual

```
usage: writeBlockMesh.py [-h] -i  -t  -o

Block cylindrical meshing

options:
  -h, --help          show this help message and exit
  -i , --input_file   Input file for meshing and geometry parameters
  -t , --topo_file    Block description of the configuration
  -o , --out_folder   Output folder for blockMeshDict

```

