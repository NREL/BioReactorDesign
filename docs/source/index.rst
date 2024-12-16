Bio Reactor Design (BiRD)
===================================

`BiRD` is a toolbox aimed a facilitating the numerical simulation of bioreactors with OpenFOAM.
The purpose of `BiRD` is 

1. to ensure reproducibility of numerical simulations of bioreactors.
2. to create a suite of test cases that can be used to test different reactor configurations.
3. to facilitate validation of multiphase flow models against several experimental campaigns.
4. to facilitate optimization (geometry and operating conditions) of bioreactors. 

It contains a python module ``bird`` that can be used to generate input files that may be read by OpenFOAM to generate meshes and cases. It can also be used to post process the output of OpenFOAM simulations.

We also provide a solver ``birdmultiphaseEulerFoam`` that contains custom models added to the base OpenFOAM-v9. 


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
