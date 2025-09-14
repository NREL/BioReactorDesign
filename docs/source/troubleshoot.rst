Troubleshoot
=====

Activate debug logs
------------

When troubleshooting issues in BiRD, we recommend making sure to switch the log level to ``DEBUG``
At the top of your script

.. code-block:: python

   from bird import logger
   logger.setLevel(level="DEBUG")


Unrecognized drag and mass transfer model
------------

.. code-block:: console

   [6] --> FOAM FATAL ERROR: 
   [6] Unknown dragModelType type Grace


This may mean that you are using ``multiphaseEulerFoam`` instead of ``birdmultiphaseEulerFoam``



BDOFoam does not compile
------------

``birdmultiphaseEulerFoam`` requires OpenFOAM 9 but ``bdoFoam`` requires OpenFOAM 6. 
Compiling ``bdoFoam`` also requires a little more work that ``birdmultiphaseEulerFoam``.
Detailed step are discussed `here <https://github.com/NREL/BioReactorDesign/issues/32>`_ 

In the future, we will make sure that ``bdoFoam`` works with OpenFOAM 9.
