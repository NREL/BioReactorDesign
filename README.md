# <ins>Bi</ins>o <ins>R</ins>eactor <ins>D</ins>esign (BiRD) Toolbox [![bird-CI](https://github.com/NREL/BioReactorDesign/actions/workflows/ci.yml/badge.svg)](https://github.com/NREL/BioReactorDesign/actions/workflows/ci.yml) [![bird-pyversion](https://img.shields.io/pypi/pyversions/NREL-bird.svg)](https://pypi.org/project/NREL-bird/)  [![bird-pypi](https://badge.fury.io/py/nrel-bird.svg)](https://badge.fury.io/py/nrel-bird)

## Installation for developers

```bash
conda create --name bird python=3.10
conda activate bird
git clone https://github.com/NREL/BioReactorDesign.git
cd BioReactorDesign
pip install -e .
```

## Installation for users

```bash
conda create --name bird python=3.10
conda activate bird
pip install nrel-bird
```

## OpenFOAM solvers

Place the attached models in `OFsolvers` into the same arborescence in your `$FOAM_APP` directory. These compile with `openFOAM-9`

We provide a new drag model `Grace`, a new interfacial composition model `Higbie` and various other models which magnitude can be controlled via an efficiency factor `*_limited`


## Meshing

### Generate Stir tank mesh

```bash
inp=bird/meshing/stir_tank_mesh_templates/base_tank/tank_par.yaml
out=bird/meshing/stir_tank_case_templates/base/system/blockMeshDict

python applications/write_stir_tank_mesh.py -i $inp -o $out
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
  <img src="https://raw.githubusercontent.com/NREL/BioReactorDesign/main/assets/stir_tank.png" width="350"/>
</p>


### Generate STL mesh

`python applications/write_stl_mesh.py -v -cr 0.25 -na 12 -aw 0.1 -al 0.5`

Generates

<p float="left">
  <img src="https://raw.githubusercontent.com/NREL/BioReactorDesign/main/assets/simpleOutput.png" width="350"/>
</p>


### Manual

```
usage: write_stl_mesh.py [-h] [-cr] [-na] [-aw] [-al] [-v]

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
## Postprocess

### Perform early prediction

`python applications/earlyPredicition.py -df bird/postProcess/data_early`

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

`python applications/compute_conditional_mean.py -f bird/postProcess/data_conditional_mean -avg 2`

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

SWR 24-35

To cite BioReactorDesign use these articles on [CO2 interphase mass transfer](https://arxiv.org/pdf/2404.19636) on [aerobic bioreactors](https://www.sciencedirect.com/science/article/pii/S0263876218304337) 
and on [butanediol synthesis](https://www.sciencedirect.com/science/article/pii/S0263876223004689)
```
@article{hassanaly2024inverse,
  title={Inverse modeling of bubble size dynamics for interphase mass transfer and gas holdup in CO2 bubble column reactors},
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



