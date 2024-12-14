# <ins>Bi</ins>o <ins>R</ins>eactor <ins>D</ins>esign (BiRD) Toolbox [![bird-CI](https://github.com/NREL/BioReactorDesign/actions/workflows/ci.yml/badge.svg)](https://github.com/NREL/BioReactorDesign/actions/workflows/ci.yml) [![bird-pyversion](https://img.shields.io/pypi/pyversions/NREL-bird.svg)](https://pypi.org/project/NREL-bird/)  [![bird-pypi](https://badge.fury.io/py/nrel-bird.svg)](https://badge.fury.io/py/nrel-bird)

## Quick start
1. Follow the steps to install the python package (see `Installation of python package for developers` or `Installation of python package for users` below)
2. Follow the steps to install the BiRD OpenFOAM solver (see `Installation of BiRD OpenFOAM solver (for developers and users)` below)
3. Check that you can run any of the tutorial cases, for ex:

```bash
cd tutorial_cases/bubble_column_20L
bash run.sh
``` 

## Installation of python package for developers

```bash
conda create --name bird python=3.10
conda activate bird
git clone https://github.com/NREL/BioReactorDesign.git
cd BioReactorDesign
pip install -e .
```

## Installation of python package for users

```bash
conda create --name bird python=3.10
conda activate bird
pip install nrel-bird
```

## Installation of BiRD OpenFOAM solver (for developers and users)

1. Activate your OpenFOAM-9 environment (`source <OpenFOAM-9 installation directory>/etc/<your-shell>rc`)
2. cd `OFsolvers/birdmultiphaseEulerFoam/`
3. `./Allwmake`

The same steps are done in the `ci.yml` (under `Test-OF - Compile solver`) which can be used as a reference. 
However, note that `ci.yml` compiles the solver in debug mode which is not suitable for production.

We provide a new drag model `Grace`, a new interfacial composition model `Higbie` and various other models which magnitude can be controlled via an efficiency factor (see [this paper](https://arxiv.org/pdf/2404.19636) for why efficiency factors are useful).


## Meshing

### Generate Stir tank mesh

```bash
inp=bird/meshing/stirred_tank_mesh_templates/base_tank/tank_par.yaml
out=bird/meshing/stirred_tank_case_templates/base/system/blockMeshDict

python applications/write_stirred_tank_mesh.py -i $inp -o $out
```

Generates a blockMeshDict

Then activate openFoam environment (tested with OpenFoam9) and mesh with

```bash
blockMesh -dict system/blockMeshDict
stitchMesh -perfect -overwrite inside_to_hub inside_to_hub_copy
stitchMesh -perfect -overwrite hub_to_rotor hub_to_rotor_copy
transformPoints "rotate=((0 0 1)(0 1 0))":
```
Mesh visualized in Paraview

<p float="left">
  <img src="https://raw.githubusercontent.com/NREL/BioReactorDesign/main/assets/stirred_tank.png" width="350"/>
</p>

#### Related tutorial

`tutorial_cases/stirred_tank`


### Block cylindrical meshing

Generates `blockMeshDict` in `system`

```bash
root=`pwd`
caseFolder=bird/meshing/block_cyl_cases_templates/case
mesh_temp=bird/meshing/block_cyl_mesh_templates/sideSparger

python applications/write_block_cyl_mesh.py -i $mesh_temp/input.json -t $mesh_temp/topology.json -o $caseFolder/system
```

Then activate openFoam environment (tested with OpenFoam9) and mesh with

```bash
cd $caseFolder
blockMesh
transformPoints "scale=(0.001 0.001 0.001)"
transformPoints "rotate=((0 0 1) (0 1 0))"
cd $root
```

Will generate this

<p float="left">
  <img src="https://raw.githubusercontent.com/NREL/BioReactorDesign/main/assets/3dsparger.png" width="250"/>
</p>


#### How to change the dimensions or mesh refinement

All dimensions and mesh are controlled by the input file `input.json`. 
The input file can also be in `.yaml` format. The parser will decide the file format based on its extension. 
See `bird/meshing/block_cyl_mesh_templates/baseColumn/` for an example of `.yaml`

#### How to change the arrangement of concentric cylinders

The block topology is controlled by the `topology.json`
Always work with a schematic. Here is the schematic for this case

<p float="left">
  <img src="https://raw.githubusercontent.com/NREL/BioReactorDesign/main/assets/schematic.png" width="250"/>
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

#### How to change boundaries

Boundaries are defined with three types, `top`, `bottom` and `lateral`

In the case of sparger walls shown below with the red lines
<p float="left">
  <img src="https://raw.githubusercontent.com/NREL/BioReactorDesign/main/assets/schematicSpargerWalls.png" width="250"/>
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
  <img src="https://raw.githubusercontent.com/NREL/BioReactorDesign/main/assets/schematicSpargerInlet.png" width="250"/>
</p>

the boundary is defined in the `topology.json` as
```
"Boundary": {
                "inlet": [
                           {"type": "lateral", "Rmin": 2, "Rmax": 3, "Lmin": 2, "Lmax": 2}
                         ],
...
```

#### Manual

```
usage: write_block_cyl_mesh.py [-h] -i  -t  -o

Block cylindrical meshing

options:
  -h, --help            show this help message and exit
  -i , --input_file     Input file for meshing and geometry parameters
  -t , --topo_file      Block description of the configuration
  -o , --output_folder 
                        Output folder for blockMeshDict
```

#### Related tutorials

- `tutorial_cases/side_sparger`
- `tutorial_cases/bubble_column_20L` 


### Block rectangular meshing

Generates `blockMeshDict` in `system`

```bash
root=`pwd`
caseFolder=bird/meshing/block_rect_cases_templates/case
mesh_temp=bird/meshing/block_rect_mesh_templates/loopReactor

python applications/write_block_rect_mesh.py -i $mesh_temp/input.json -o $caseFolder/system
```

Then activate openFoam environment (tested with OpenFoam9) and mesh with

```bash
cd $caseFolder
blockMesh
cd $root
```

Will generate this

<p float="left">
  <img src="https://raw.githubusercontent.com/NREL/BioReactorDesign/main/assets/loop_react.png" width="400"/>
</p>


#### How to change the block rectangular geometry

The geometry of the block cylindrical mesh is defined within a 3D domain (X,Y,Z). The blocks that represent the fluid domain are a subset of a block rectangular background domain. The fluid blocks are defined using the geometry corners. For the mesh shown above, the geometry corners are the red blocks shown below 

<p float="left">
  <img src="https://raw.githubusercontent.com/NREL/BioReactorDesign/main/bird/meshing/block_rect_mesh_templates/loopReactor/loop_schematic.png" width="700"/>
</p>
 
The corners are defined in the `input.json`
```
"Geometry": {
        "Fluids": [
                [ [0,0,0], [9,0,0], [9,0,4], [0,0,4] ],
                [ [0,1,4], [0,4,4], [0,4,0], [0,1,0] ]
        ]
}
...
```

#### Related tutorials

- `tutorial_cases/loop_reactor_mixing`
- `tutorial_cases/loop_reactor_reacting` 

## Preprocess
### Generate STL mesh

Boundaries may be specified with `surfaceToPatch` utility in OpenFOAM, based on STL files that can be generated with 

`python applications/write_stl_patch.py -v`

Generates

<p float="left">
  <img src="https://raw.githubusercontent.com/NREL/BioReactorDesign/main/assets/simpleOutput.png" width="350"/>
</p>


To see how to use this on an actual case see `tutorial_cases/loop_reactor_mixing` and `tutorial_cases/loop_reactor_reacting` 

### Manual

```
usage: write_stl_patch.py [-h] [-i] [-v]

Generate boundary patch

options:
  -h, --help     show this help message and exit
  -i , --input   Boundary patch Json input
  -v, --verbose  plot on screen
```
### How to change the set of shapes in the boundary patch

Edit the json files read when generating the mesh. In the case below, the boundary condition `inlets` consists of 3 discs 

```
{
    "inlets": [
        {"type": "circle", "centx": 5.0, "centy": 0.0, "centz": 0.5, "radius": 0.4, "normal_dir": 1,"nelements": 50},
        {"type": "circle", "centx": 2.5, "centy": 0.0, "centz": 0.5, "radius": 0.4, "normal_dir": 1,"nelements": 50},
        {"type": "circle", "centx": 7.5, "centy": 0.0, "centz": 0.5, "radius": 0.4, "normal_dir": 1,"nelements": 50}
    ],
}
...
```
### Related tutorials

- `tutorial_cases/bubble_column_20L`
- `tutorial_cases/loop_reactor_mixing`
- `tutorial_cases/loop_reactor_reacting` 


## Postprocess

### Perform early prediction

`python applications/early_prediction.py -df bird/postprocess/data_early`

Generates

<p float="left">
  <img src="https://raw.githubusercontent.com/NREL/BioReactorDesign/main/assets/early_det.png" width="350"/>
  <img src="https://raw.githubusercontent.com/NREL/BioReactorDesign/main/assets/early_uq.png" width="350"/>
</p>

#### Manual

```
usage: early_prediction.py [-h] -df  [-func]

Early prediction

options:
  -h, --help            show this help message and exit
  -df , --dataFolder    Data folder containing multiple QOI time histories
  -func , --functionalForm 
                        functional form used to perform extrapolation
```


### Plot conditional means

`python applications/compute_conditional_mean.py -f bird/postprocess/data_conditional_mean -avg 2`

Generates (among others)

<p float="left">
  <img src="https://raw.githubusercontent.com/NREL/BioReactorDesign/main/assets/gh_cond_mean.png" width="350"/>
  <img src="https://raw.githubusercontent.com/NREL/BioReactorDesign/main/assets/co2g_cond_mean.png" width="350"/>
</p>


```bash
usage: compute_conditional_mean.py [-h] -f  [-vert] [-avg] [--fl FL [FL ...]] [-n  [...]]

Compute conditional means of OpenFOAM fields

options:
  -h, --help            show this help message and exit
  -f , --caseFolder     caseFolder to analyze
  -vert , --verticalDirection 
                        Index of vertical direction
  -avg , --windowAve    Window Average
  --fl FL [FL ...], --field_list FL [FL ...]
                        List of fields to plot
  -n  [ ...], --names  [ ...]
                        names of cases
```

<!--
## Preprocess

### Generate fi.gas

`cd inhomogeneousBC`

follow `README.md`
-->

## Formatting [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

Code formatting and import sorting are done automatically with `black` and `isort`.

Fix imports and format : `pip install black isort; bash fixFormat.sh`

Spelling is checked but not automatically fixed using `codespell`


## References

Software record [SWR 24-35](https://www.osti.gov/biblio/2319227)

To cite BioReactorDesign use these articles on [CO2 interphase mass transfer (open access)](https://arxiv.org/pdf/2404.19636) on [aerobic bioreactors](https://www.sciencedirect.com/science/article/pii/S0263876218304337) 
and on [butanediol synthesis](https://www.sciencedirect.com/science/article/pii/S0263876223004689)
```
@article{hassanaly2024inverse,
  title={Bayesian calibration of bubble size dynamics applied to \ce{CO2} gas fermenters},
  author={Hassanaly, Malik and Parra-Alvarez, John M. and Rahimi, Mohammad J. and Sitaraman, Hariswaran},
  journal={Under Review},
  year={2024},
}


@article{rahimi2018computational,
  title={Computational fluid dynamics study of full-scale aerobic bioreactors: Evaluation of gas--liquid mass transfer, oxygen uptake, and dynamic oxygen distribution},
  author={Rahimi, Mohammad J and Sitaraman, Hariswaran and Humbird, David and Stickel, Jonathan J},
  journal={Chemical Engineering Research and Design},
  volume={139},
  pages={283--295},
  year={2018},
  publisher={Elsevier}
}

@article{sitaraman2023reacting,
  title={A reacting multiphase computational flow model for 2, 3-butanediol synthesis in industrial-scale bioreactors},
  author={Sitaraman, Hariswaran and Lischeske, James and Lu, Yimin and Stickel, Jonathan},
  journal={Chemical Engineering Research and Design},
  volume={197},
  pages={38--52},
  year={2023},
  publisher={Elsevier}
}
```

## Acknowledgments

This work was authored by the National Renewable Energy Laboratory (NREL), operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308. This work was supported by funding from DOE Bioenergy Technologies Office (BETO) [CO2RUe consortium](https://www.energy.gov/eere/co2rue). The research was performed using computational resources sponsored by the Department of Energy's Office of Energy Efficiency and Renewable Energy and located at the National Renewable Energy Laboratory. The views expressed in the article do not necessarily represent the views of the DOE or the U.S. Government. The U.S. Government retains and the publisher, by accepting the article for publication, acknowledges that the U.S. Government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this work, or allow others to do so, for U.S. Government purposes.



