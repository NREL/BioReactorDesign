Tutorial for unstructured grids of tetrahedrons with rotating mesh
--------------------------------------------------------
Features demonstrated:
* 500 mL sparging oxygen in water at 1VVM
* Mass transfer
* Dynamic mesh region rotating at 100 RPM
* High Courant number schemes
* Pupulation balance model

Used for:
* Unsteady calculation in complex geometries

The base case should run with $\Delta t \approx 5e^{-4}$ s.
__Uses OpenFOAM-13. Mesh is finer than the MRF tutorial because of NCC refinement.__

__Example results__
Volume colors are gas volume fraction and surface color are liquid velocity.

![Gas volume fraction](/tutorials_13/STR_rotatingMesh/STR_rotating.gif)
  
