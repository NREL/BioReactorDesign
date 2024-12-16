Meshing
=====


.. _stir_tank:

Generate stirred tank reactor mesh
------------

Generate a ``blockMeshDict`` with

.. code-block:: console

   inp=bird/meshing/stirred_tank_mesh_templates/base_tank/tank_par.yaml
   out=bird/meshing/stirred_tank_case_templates/base/system/blockMeshDict

   python applications/write_stirred_tank_mesh.py -i $inp -o $out

Then activate openFoam environment and mesh with

.. code-block:: console

   blockMesh -dict system/blockMeshDict
   stitchMesh -perfect -overwrite inside_to_hub inside_to_hub_copy
   stitchMesh -perfect -overwrite hub_to_rotor hub_to_rotor_copy
   transformPoints "rotate=((0 0 1)(0 1 0))"

Visualize mesh in Paraview

.. _fig:stirredtank:

.. figure:: https://raw.githubusercontent.com/NREL/BioReactorDesign/main/assets/stirred_tank.png
     :width: 70%
     :align: center
     :name: fig-str
     :target: https://raw.githubusercontent.com/NREL/BioReactorDesign/main/assets/stirred_tank.png
     :alt: Stirred-tank reactor

Related tutorial
^^^^^^^^^^^^^^^

- ``tutorial_cases/stirred_tank``



.. _block_cyl:

Block cylindrical meshing
------------

Generates ``system/blockMeshDict``

.. code-block:: console

   root=`pwd`
   caseFolder=bird/meshing/block_cyl_cases_templates/case
   mesh_temp=bird/meshing/block_cyl_mesh_templates/sideSparger

   python applications/write_block_cyl_mesh.py -i $mesh_temp/input.json -t $mesh_temp/topology.json -o $caseFolder/system

Then activate openFoam environment and mesh with

.. code-block:: console

   cd $caseFolder
   blockMesh
   transformPoints "scale=(0.001 0.001 0.001)"
   transformPoints "rotate=((0 0 1) (0 1 0))"
   cd $root

Visualize mesh in Paraview

.. _fig:sidesparger:

.. figure:: https://raw.githubusercontent.com/NREL/BioReactorDesign/main/assets/3dsparger.png
     :width: 50%
     :align: center
     :name: fig-sidesparger
     :target: https://raw.githubusercontent.com/NREL/BioReactorDesign/main/assets/3dsparger.png
     :alt: Reactor with a side sparger

How to change the dimensions or mesh refinement?
^^^^^^^^^^^^^^^

All dimensions and mesh are controlled by the input file ``input.json``.
The input file can also be in ``.yaml`` format. The parser will decide the file format based on its extension.
See ``bird/meshing/block_cyl_mesh_templates/baseColumn/`` for an example of ``.yaml``

How to change the arrangement of concentric cylinders?
^^^^^^^^^^^^^^^

The block topology is controlled by the ``topology.json``
We recomment always working with a schematic. Here is the schematic for this case

.. figure:: https://raw.githubusercontent.com/NREL/BioReactorDesign/main/assets/schematic.png
     :width: 50%
     :align: center
     :name: fig-schematic
     :target: https://raw.githubusercontent.com/NREL/BioReactorDesign/main/assets/schematic.png
     :alt: Side sparger schematic


The purple blocks are walls (not meshed) and the white blocks are fluid blocks (meshed). The symmetry axis is indicated as a dashed line

In the ``topology.json``, the purple blocks are defined as

.. code-block:: json

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


How to change boundaries?
^^^^^^^^^^^^^^^

Boundaries are defined with three types, ``top``, ``bottom`` and ``lateral``

In the case of sparger walls shown below with the red lines

.. figure:: https://raw.githubusercontent.com/NREL/BioReactorDesign/main/assets/schematicSpargerWalls.png
     :width: 50%
     :align: center
     :name: fig-schematicwalls
     :target: https://raw.githubusercontent.com/NREL/BioReactorDesign/main/assets/schematicSpargerWalls.png
     :alt: Wall side sparger schematic


the boundary is defined in the ``topology.json`` as

.. code-block:: json

   "Boundary": {
                   "wall_sparger":[
                              {"type": "bottom", "Rmin": 2, "Rmax": 2, "Lmin": 2, "Lmax": 3},
                              {"type": "top", "Rmin": 0, "Rmax": 0, "Lmin": 1, "Lmax": 2},
                              {"type": "top", "Rmin": 1, "Rmax": 1, "Lmin": 1, "Lmax": 2},
                              {"type": "top", "Rmin": 2, "Rmax": 2, "Lmin": 1, "Lmax": 2}
                            ],

For the side sparger, the inlet is shown below with the red line

.. figure:: https://raw.githubusercontent.com/NREL/BioReactorDesign/main/assets/schematicSpargerInlet.png
     :width: 50%
     :align: center
     :name: fig-schematicinlet
     :target: https://raw.githubusercontent.com/NREL/BioReactorDesign/main/assets/schematicSpargerInlet.png
     :alt: Inlet side sparger schematic

the boundary is defined in the ``topology.json`` as

.. code-block:: json

   "Boundary": {
                   "inlet": [
                              {"type": "lateral", "Rmin": 2, "Rmax": 3, "Lmin": 2, "Lmax": 2}
                            ],



Related tutorials
^^^^^^^^^^^^^^^

- ``tutorial_cases/side_sparger``
- ``tutorial_cases/bubble_column_20L``


Block rectangular meshing
------------

Generates ``system/blockMeshDict``

.. code-block:: console

   root=`pwd`
   caseFolder=bird/meshing/block_rect_cases_templates/case
   mesh_temp=bird/meshing/block_rect_mesh_templates/loopReactor

   python applications/write_block_rect_mesh.py -i $mesh_temp/input.json -o $caseFolder/system

Then activate openFoam environment and mesh with

.. code-block:: console

   cd $caseFolder
   blockMesh
   cd $root

Visualize mesh in Paraview

.. _fig:loop_reactor:

.. figure:: https://raw.githubusercontent.com/NREL/BioReactorDesign/main/assets/loop_react.png
     :width: 80%
     :align: center
     :name: fig-loopreactor
     :target: https://raw.githubusercontent.com/NREL/BioReactorDesign/main/assets/loop_react.png
     :alt: Loop reactor


How to change the block rectangular geometry?
^^^^^^^^^^^^^^^
The geometry of the block cylindrical mesh is defined within a 3D domain (X,Y,Z). The blocks that represent the fluid domain are a subset of a block rectangular background domain. The fluid blocks are defined using the geometry corners. For the mesh shown above, the geometry corners are the red blocks shown below

.. _fig:loop_reactor_schematic:

.. figure:: https://raw.githubusercontent.com/NREL/BioReactorDesign/main/bird/meshing/block_rect_mesh_templates/loopReactor/loop_schematic.png
     :width: 80%
     :align: center
     :name: fig-loopreactor-schematic
     :target: https://raw.githubusercontent.com/NREL/BioReactorDesign/main/bird/meshing/block_rect_mesh_templates/loopReactor/loop_schematic.png
     :alt: Loop reactor schematic


The corners are defined in the ``input.json``

.. code-block:: json

   "Geometry": {
           "Fluids": [
                   [ [0,0,0], [9,0,0], [9,0,4], [0,0,4] ],
                   [ [0,1,4], [0,4,4], [0,4,0], [0,1,0] ]
           ]
   }


Related tutorials
^^^^^^^^^^^^^^^
- ``tutorial_cases/loop_reactor_mixing``
- ``tutorial_cases/loop_reactor_reacting``


