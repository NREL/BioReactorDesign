Bio Reactor Design (BiRD)
===================================

`BiRD` is a toolbox for computational fluid dynamics (CFD) simulations of bioreactors with OpenFOAM.

The objective of `BiRD` is to

1. ensure reproducibility of numerical simulations of bioreactors.
2. create a centralized suite of test cases for different reactor configurations.
3. facilitate validation of multiphase flow models against several experimental campaigns.
4. facilitate optimization (geometry and operating conditions) of bioreactors. 

It contains a python module ``bird`` that can be used to generate input files that may be read by OpenFOAM to generate meshes and cases. It can also be used to post process the output of OpenFOAM simulations.

We provide a solver ``birdmultiphaseEulerFoam`` that contains custom models added to the base OpenFOAM-v9. 


.. note::

   This project is under active development. We welcome community contributions and testing!


.. toctree::
   :hidden:

   quickstart
   meshing
   preprocess
   postprocess
   contribute
   references
   acknowledgments
