Quick start
=====


.. _installation_dev:

Installation of python package for developers
------------


.. code-block:: console

   conda create --name bird python=3.10
   conda activate bird
   git clone https://github.com/NREL/BioReactorDesign.git
   cd BioReactorDesign
   pip install -e .

.. _installation_users:

Installation of python package for users
------------


.. code-block:: console

   conda create --name bird python=3.10
   conda activate bird
   pip install nrel-bird


.. _installation_of:

Installation of BiRD OpenFOAM solver (for developers and users)
------------

1. Activate your OpenFOAM-9 environment 

.. code-block:: console

   source <OpenFOAM-9 installation directory>/etc/<your-shell>rc

2. Compile the solver

.. code-block:: console

   cd OFsolvers/birdmultiphaseEulerFoam/
   ./Allwmake



Run an example
----------------

1. Follow the steps to install the python package (see either the :ref:`Installation section for developers<installation_dev>` or the :ref:`Installation section for users<installation_users>`)
2. Follow the steps to install the BiRD OpenFOAM solver (see the :ref:`Installation section for the solver<installation_of>`) 
3. Check that you can run any of the tutorial cases, for example

.. code-block:: console

   cd tutorial_cases/bubble_column_20L
   bash run.sh

