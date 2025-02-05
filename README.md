# <ins>Bi</ins>o <ins>R</ins>eactor <ins>D</ins>esign (BiRD) [![bird-CI](https://github.com/NREL/BioReactorDesign/actions/workflows/ci.yml/badge.svg)](https://github.com/NREL/BioReactorDesign/actions/workflows/ci.yml) [![bird-pyversion](https://img.shields.io/pypi/pyversions/NREL-bird.svg)](https://pypi.org/project/NREL-bird/)  [![bird-pypi](https://badge.fury.io/py/nrel-bird.svg)](https://badge.fury.io/py/nrel-bird)

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
2. `cd OFsolvers/birdmultiphaseEulerFoam/`
3. Compile `./Allwmake`

## Documentation

See the [nrel.github.io/BioReactorDesign](https://nrel.github.io/BioReactorDesign).


## References

Software record [SWR 24-35](https://www.osti.gov/biblio/2319227)

To cite BiRD, please use these articles on [CO2 interphase mass transfer](https://doi.org/10.1016/j.cherd.2025.01.034) (open access [link](https://arxiv.org/pdf/2404.19636) ) on [aerobic bioreactors](https://doi.org/10.1016/j.cherd.2018.08.033) and on [butanediol synthesis](https://doi.org/10.1016/j.cherd.2023.07.031)


```
@article{hassanaly2025inverse,
    title={Bayesian calibration of bubble size dynamics applied to \ce{CO2} gas fermenters},
    author={Hassanaly, Malik and Parra-Alvarez, John M. and Rahimi, Mohammad J., Municchi, Federico and Sitaraman, Hariswaran},
    journal={Chemical Engineering Research and Design},
    year={2025},
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



