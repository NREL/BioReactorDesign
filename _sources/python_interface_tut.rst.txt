Post processing with the python interface to OpenFOAM
=====

This tutorial demonstrates how to post process a case with the python interface to OpenFOAM provided in BiRD.

The tutorial assumes you have created and activated the ``bird`` environment and setup path variables as

.. code-block:: console

   conda activate bird
   BIRD_HOME=`python -c "import bird; print(bird.BIRD_DIR)"`
   DCM_CASE=${BIRD_HOME}/postprocess/data_conditional_mean/
   cd $DCM_CASE

The case and the data correspond to a coflowing bubble column.

This tutorial is code-along, meaning to that you can execute the commands listed in the code-blocks. You can also execute the entirety of the tutorial in ``$DCM_CASE/python_interface_tut.py``

Reading fields
------------

The mesh manipulation tools are located in ``bird.utilities.ofio``

Before starting to manipulate the fields with the python interface, we need mesh descriptor object files, in particular ``meshCellCentres_1.obj``. This file contains the coordinates of the cell centers which is the location where the internal fields values are. The folder ``$DCM_CASE`` contains pre-generated mesh descriptor object files. The user can generate these files with ``writeMeshObj -case {case_folder}``. This command will not work for this tutorial as it would require activating your OpenFOAM environment and would require a ``constant`` and ``system`` folders that we did not add because of space constraints.

To read the cells centers

.. code-block:: python

   from bird.utilities.ofio import *   
   cell_centers = readMesh("meshCellCentres_1.obj") 

``cell_centers`` is a :math:`(N,3)` numpy array that contains the cell center coordinates (:math:`N` is the number of cells in the domain)


If we want to read the gas volume fraction ``alpha.gas``, the mass fraction of CO2 in the gas phase ``CO2.gas``, and the liquid velocity at time ``80/`` we can run

.. code-block:: python

   co2_gas = readOF("80/CO2.gas")
   alpha_gas = readOF("80/alpha.gas")
   u_liq = readOF("80/U.liquid")
   print("cell CO2 gas shape = ", co2_gas["field"].shape)
   print("cell alpha gas shape = ", alpha_gas["field"].shape)
   print("cell u liq shape = ", u_liq["field"].shape)

The function ``readOF`` generates a dictionary of values and automatically detects whether the field is a vector or a scalar field.



Plot conditional means
------------

With cells center locations and the internal fields, one can use python functions to post process the case. For example, we might want to compute conditional averages over a spatial direction, say y

.. code-block:: python
   
   from bird.utilities.mathtools import conditional_average

   y_co2_gas_cond, co2_gas_cond = conditional_average(
       cell_centers[:, 1], co2_gas["field"], nbins=32
   )
   y_alpha_gas_cond, alpha_gas_cond = conditional_average(
       cell_centers[:, 1], alpha_gas["field"], nbins=32
   )
   
   from prettyPlot.plotting import *
   fig = plt.figure()
   plt.plot(y_co2_gas_cond, co2_gas_cond, color="k", label=r"$Y_{CO_2}$ [-]")
   plt.plot(
       y_alpha_gas_cond, alpha_gas_cond, color="b", label=r"$\alpha_{g}$ [-]"
   )
   pretty_labels("Y [m]", "", fontsize=20, grid=False, fontname="Times")
   pretty_legend(fontname="Times")
   plt.show()


This will show the following plot


.. container:: figures-cond-mean-tut

   .. figure:: ../assets/cond_mean_tut.png
      :width: 70%
      :align: center
      :alt: Height-conditional mean


Compute reactor properties
------------

The python interface is also useful to compute reactor averaged properties. We usually like to compute volume averaged properties, which requires access to the cell volume. A cell volume field ``V`` can be written using OpenFOAM utilities (``postProcess -func writeCellVolumes -time {time_folder} -case {case_folder}``). Running this command would again require activating the OpenFOAM environment and we already provide a volume field in the ``1/`` folder here. 

A typical example is that one would want to compute at time 80

1. gas hold up (``gh``)
2. superficial velocity (``sup_vel``)
3. reactor volume averaged mass fraction of CO2 in the liquid phase (``y_ave_co2``)
4. reactor volume averaged concentration of CO2 in the liquid phase (``c_ave_co2``)
5. Reactor averaged bubble diameter (``diam``)

Several of these quantities, will require reading and processing the same fields. For example, both ``y_ave_co2`` and ``c_ave_co2`` require to read ``CO2.liquid``. To avoid re-reading the same fields, we store the fields in ``field_dict`` that allows to reuse fields when possible.

.. code-block:: python

   from bird.postprocess.post_quantities import *

   # Compute Gas hold up
   kwargs = {"case_folder": ".", "time_folder": "80"}
   gh, field_dict = compute_gas_holdup(
       volume_time="1", field_dict={"cell_centers": cell_centers}, **kwargs
   )
   print("fields stored = ", list(field_dict.keys()))
   print(f"Gas Holdup = {gh:.4g}")
   
   # Compute superficial velocity
   sup_vel, field_dict = compute_superficial_velocity(
       volume_time="1", field_dict=field_dict, **kwargs
   )
   print("fields stored = ", list(field_dict.keys()))
   print(f"Superficial velocity = {sup_vel:.4g} m/s")
   
   # Compute reactor-averaged CO2 mass fraction
   y_ave_co2, field_dict = compute_ave_y_liq(
       volume_time="1", spec_name="CO2", field_dict=field_dict, **kwargs
   )
   print("fields stored = ", list(field_dict.keys()))
   print(f"Reactor averaged YCO2 = {y_ave_co2:.4g}")
   
   # Compute reactor-averaged CO2 concentration
   c_ave_co2, field_dict = compute_ave_conc_liq(
       volume_time="1",
       spec_name="CO2",
       mol_weight=0.04401,
       rho_val=1000,
       field_dict=field_dict,
       **kwargs,
   )
   print("fields stored = ", list(field_dict.keys()))
   print(f"Reactor averaged [CO2] = {c_ave_co2:.4g} mol/m3")
   
   # Compute reactor-averaged bubble diameter
   diam, field_dict = compute_ave_bubble_diam(
       volume_time="1", field_dict=field_dict, **kwargs
   )
   print("fields stored = ", list(field_dict.keys()))
   print(f"Reactor averaged bubble diameter = {diam:.4g} m")
   


This should generate the following 


.. code-block:: console

   fields stored =  ['cell_centers', 'alpha.liquid', 'V']
   Gas Holdup = 0.3041
   fields stored =  ['cell_centers', 'alpha.liquid', 'V', 'alpha.gas', 'U.gas']
   Superficial velocity = 0.08241 m/s
   fields stored =  ['cell_centers', 'alpha.liquid', 'V', 'alpha.gas', 'U.gas', 'CO2.liquid', 'ind_liq']
   Reactor averaged YCO2 = 0.0002948
   fields stored =  ['cell_centers', 'alpha.liquid', 'V', 'alpha.gas', 'U.gas', 'CO2.liquid', 'ind_liq', 'rho_liq']
   Reactor averaged [CO2] = 6.698 mol/m3
   fields stored =  ['cell_centers', 'alpha.liquid', 'V', 'alpha.gas', 'U.gas', 'CO2.liquid', 'ind_liq', 'rho_liq', 'd.gas']
   Reactor averaged bubble diameter = 0.008497 m


The ``fields stored`` print shows what fields are read to compute each quantity. Between the calculation of ``y_ave_co2`` and ``c_ave_co2``, only ``rho_liq`` was added to the list of fields read. In other terms, the parser recycled ``CO2.liquid`` instead of re-reading it. Obviously, this approach trades input/output operations for memory use and it is up to the user to decide what is the right approach. In this case, reusing the fields reduces the computational cost by about 50% (0.26s when not reusing the fields, 0.17s when reusing the fields on an M1 Mac). 
