# <ins>Bi</ins>o <ins>R</ins>eactor <ins>D</ins>esign (BiRD) [![bird-CI](https://github.com/NatLabRockies/BioReactorDesign/actions/workflows/ci.yml/badge.svg)](https://github.com/NatLabRockies/BioReactorDesign/actions/workflows/ci.yml) [![bird-pyversion](https://img.shields.io/pypi/pyversions/NLR-bird.svg)](https://pypi.org/project/NLR-bird/) [![coverage](https://codecov.io/gh/NatlabRockies/BioReactorDesign/graph/badge.svg)](https://app.codecov.io/gh/natlabrockies/bioreactordesign)  [![bird-pypi](https://badge.fury.io/py/nlr-bird.svg)](https://badge.fury.io/py/nlr-bird)

## Quick start
1. Follow the steps to install the python package (see `Installation of python package for developers` or `Installation of python package for users` below)
2. Follow the steps to install the BiRD OpenFOAM solver (see `Installation of BiRD OpenFOAM solver (for developers and users)` below)
3. Check that you can run any of the tutorial cases, for ex:

```bash
cd tutorial_cases/bubble_column_20L
bash run.sh
``` 

## Installation of python package for developers (recommended)

```bash
conda create -c conda-forge --name bird python=3.14 paraview
conda activate bird
git clone https://github.com/NatLabRockies/BioReactorDesign.git
cd BioReactorDesign
pip install -e .
```

## Installation of python package for developers (with `pixi`)
This requires installing `pixi`. If it is not available on your machine, follow [Pixi Installation](https://pixi.prefix.dev/latest/installation).
Once installed, `pixi` does not require the `conda` environment creation and `pip install` gymnastics, and can be used as

```bash
git clone https://github.com/NatLabRockies/BioReactorDesign.git
cd BioReactorDesign
pixi shell
```
This will activate an environment for you (equivalent to `pip install -e .`)
Alternatively to the `pixi shell`, in any directory of BiRD, you can use `pixi run python myscript.py` instead of `python myscript.py`.


## Installation of python package for users

```bash
conda create -c conda-forge --name bird python=3.14 paraview
conda activate bird
pip install nlr-bird=={version}
```

We highly recommend that you specify the version explicitly as the library is still rapidly changing.

## Installation of BiRD OpenFOAM solver (for developers and users)

1. Activate your OpenFOAM environment (`source <OpenFOAM installation directory>/etc/<your-shell>rc`)
2. For OpenFOAM-9: `cd OFsolvers/OF9/birdmultiphaseEulerFoam/`. For OpenFOAM-13: `cd OFsolvers/OF9/birdmultiphaseEulerFoam/`
3. Compile `./Allwmake`

## Documentation

See the [natlabrockies.github.io/BioReactorDesign](https://natlabrockies.github.io/BioReactorDesign).


## References

Software record [SWR 24-35](https://www.osti.gov/biblio/code-123482)

To cite BiRD, please refer to [references](https://natlabrockies.github.io/BioReactorDesign/references.html)


## Acknowledgments

This work was authored in part by the National Laboratory of the Rockies (NLR) for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308. This work was supported by funding from DOE Alternative Fuels and Feedstocks Office (AFFO) and Fossil Energy (FE) office through the [CO2RUe consortium](https://www.energy.gov/eere/co2rue). The research was performed using computational resources sponsored by the Department of Energy's Office of Critical Minerals and Energy Innovation (CMEI) and located at the National Laboratory of the Rockies. The views expressed in the article do not necessarily represent the views of the DOE or the U.S. Government. The U.S. Government retains and the publisher, by accepting the article for publication, acknowledges that the U.S. Government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this work, or allow others to do so, for U.S. Government purposes.



