Python interface to OpenFOAM
=====

We provide a simple python interface for reading OpenFOAM case results and dictionaries.
A more comprehensive interface is available through ``pyFoam`` although we found it difficult to use and without recent support, which motivated the implementation of a new python interface.  


Read fields
------------

Internal scalar and vector fields
~~~~~~~~~~~~~~~~~~~~

Currently internal scalar and vector fields can be read using the python interface. In particular, note that 
#. 1. We do not support reading tensor fields for now.
#. 2. We do not support reading boundary fields for now.
#. 3. We only read reconstructed files

We are open to implementing 1. and 2. if need be. Implementing 3. is possible but will be more involved.

The main function interface to read openFOAM fields is ``readOF`` in ``bird.utilities.ofio``

The function only reads fields written in ASCII format and decided based on the header whether the field is a scalar or a vector. In the case of scalar, ``readOF`` returns a (N,) numpy array, where N is the number of computational cells. In the case of a vector, ``readOF`` returns a (N,3) numpy array.

If a uniform field is read, the number of cells may not be available from the field file and the function returns a float (equal to the uniform internal field value). If a numpy array is needed, the user can specify the number of cells in the field as an input.

Reuse instead of re-read
~~~~~~~~~~~~~~~~~~~~

The ``read_field`` function uses ``readOF`` and takes a dictionary ``field_dict`` as input which is used to avoid reading multiple times the same field. For example, if one wants to compute the reactor-averaged concentration of a species, and then the reactor-averaged mass fraction of a species, the same mass fraction field will be used in both cases. As fields are read, ``field_dict`` will store the mass fraction field and recognize that the same field is needed.

It is up to the user to reinitialize ``field_dict``. For example, if the reactor-averaged mass fraction needs to be computed at time ``1`` and then at time ``2``, the user needs to pass an empty dictionary (or nothing) to ``read_field`` before reading the fields at time ``2``. Otherwise, ``read_field`` will assume that the mass fraction field is the same between time ``1`` and time ``2``.


Read mesh-related fields
~~~~~~~~~~~~~~~~~~~~

We rely on OpenFOAM utilities to provide mesh-based fields. The results of the OpenFOAM utilities can still be processed using ``bird`` functions.


Reading cell volumes 
^^^^^^^^^^^^^^^

A cell volume field can be generated using the following OpenFOAM command ``postProcess -func writeCellVolumes -time {time_folder} -case {case_folder}``
It will generate a file ``{time_folder}/V`` which can be read with the ``readOF`` function of ``bird``.
This workflow is used in ``bird.postprocess.post_quantitities``, for example in the ``compute_gas_holdup`` function.

 
Reading cell centers 
^^^^^^^^^^^^^^^

A mesh object file can be generated with the OpenFOAM command  ``writeMeshObj -case {case_folder}``
The file can then be read with the function ``readMesh`` from ``bird.utilities.ofio``. 
Again, this is used in ``bird.postprocess.post_quantities`` in the ``compute_superficial_velocity`` function.




Read dictionaries
------------

We provide a function ``parse_openfoam_dict`` in ``bird.utilities.ofio`` that can parse OpenFOAM dictionaries. The function requires a lot of special characters handling but works for processing basic dictionaries needed to manage OpenFOAM cases (``controlDict``, ``setFieldsDict``, ``phaseProperties``, ``thermophysicalProperties``, ``momentumTransport``, ...)


Generate cases
------------

(to be added based on the reactor optimization work)
