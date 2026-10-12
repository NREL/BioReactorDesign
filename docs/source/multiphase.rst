Multiphase flow model
=====


Multiphase transport equations
------------

The multiphase flow model uses an Euler-Euler approach (already available in ``multiphaseEulerFoam``).
Gas and liquid volume fractions are transported as 

 .. math::

   \frac{\partial (\alpha_j \rho_j)}{\partial t} + \nabla \cdot (\alpha_j \rho_j u_j) = \sum_{k\neq j} \dot{m}_{kj},
  
where :math:`j` is the phase index (gas or liquid), :math:`\alpha_j` is the volume fraction of the phase :math:`j`, :math:`\rho_j` is the density of phase :math:`j`, :math:`u_j` is the velocity vector of phase :math:`j`. The right-hand side of this equation includes the sum of interphase mass transfer terms where :math:`\dot{m}_{kj}` is the mass transfer rate from phase :math:`k` to phase :math:`j`. To model the relative motion of each phase, a phase-momentum transport equation is solved, which takes the form

 .. math::

   \frac{\partial (\alpha_j \rho_j u_j)}{\partial t} + \nabla \cdot (\alpha_j \rho_j u_j u_j) = \nabla \cdot (\alpha_j \overline{\tau}) - \alpha_j \nabla p + \alpha_j \rho_j g +\sum_{k\neq j} D_{kj} + M^{m}_j + F_j,

where :math:`\overline{\tau}` is the stress tensor which includes Reynolds stress and viscous (molecular and turbulent) stress (including turbulent viscosity) for the phase :math:`j`; :math:`D_{kj}` is the drag force exerted by phase :math:`k` on phase :math:`j` that depends on the drag coefficient associated with the dispersed phase and associated volume fraction, and :math:`F_j` contains interfacial forces acting on phase :math:`j` which includes lift, wall-lubrication, virtual-mass and turbulent-dispersion forces. :math:`M^{m}_j` accounts for the added momentum and is defined as :math:`M^{m}_j = \dot{m}_{j,in} u_k - \dot{m}_{j,out} u_j` where :math:`\dot{m}` is the mass transfer rate from/to phase :math:`j`, :math:`u_j` is the velocity of phase :math:`j` and :math:`u_k` is the velocity of phase :math:`k`.

If needed, an energy equation is solved to account for differences in phase temperatures.
    
 .. math::
   \frac{\partial \rho_j \alpha_j E_j}{\partial t} + \nabla \cdot (\rho_j \alpha_j u_j E_j) = \nabla \cdot (\alpha_j \kappa_{j} \nabla T_j) + \dot{Q}

where :math:`E_j` is the sensible energy of the phase :math:`j`, :math:`T_j` is the temperature of the phase :math:`j`, :math:`\kappa_{j}` is the effective thermal diffusivity of the phase :math:`j` (including molecular and turbulent contributions) and :math:`\dot{Q}` is the heat exchange due to differences in temperature. Heat exchange through the interface is driven by temperature differences between phases. The last right-hand side term can be written as :math:`\dot{Q} = h_{jl}(T_f - T_j)`, where :math:`h_{jl}` is the heat transfer coefficient of species :math:`l` in phase :math:`j`, :math:`T_j` is the temperature of phase :math:`j` and :math:`T_f` is the temperature at the interface. The temperature at the interface is computed based on the assumption that the rate of heat transfer must be equal to the latent heat :math:`\lambda_j` at the interface between phases :math:`j` and :math:`k` where :math:`h_{jl}(T_j - T_f) + h_{kl}(T_k - T_f) = \dot{m}_{j,in} \lambda_j`

Using the momentum and phase fraction, species mass fractions are transported as

 .. math::
    \frac{\partial (\rho_j \alpha_j Y_{jl})}{\partial t} + \nabla \cdot (\rho_j \alpha_j Y_{jl} u_j) = \nabla \cdot (\rho_j \alpha_j D_{jl} \nabla Y_{jl}) + S_{jl},

where :math:`Y_{jl}` is the :math:`l^{th}` species mass fraction in phase :math:`j`, :math:`D_{jl}` is the diffusivity of the :math:`l^{th}` species in phase :math:`j` and :math:`S_{jl}` are source terms due to interphase mass transfer.

If desired, a population balance equation can be solved to model the distribution of bubble size. The fundamental governing equation of the NDF is

 .. math::
    \frac{\partial n_v}{\partial t} + \nabla \cdot (u n_v) = h_v,

where $:math:`u` is the phase velocity, :math:`t` is time, :math:`n_v` is the number density of bubbles of size :math:`v` and the source term is :math:`h_v =  B_{\rm b} - B_{\rm d} +  C_{\rm b} -  C_{\rm d} - D + \dot{N_v}`, where :math:`B_{\rm b}` (resp. :math:`C_{\rm b}`) is the bubble birth contribution from bubble breakup (resp. coalescence), :math:`B_{\rm d}` (resp. :math:`C_{\rm d}`) is the bubble death contribution from bubble breakup (resp. coalescence), :math:`D` is the drift term that is due to changes of bubble volume arising from pressure or mass transfer induced density changes, and :math:`\dot{N_v}` is the bubble nucleation source term. The full expression of :math:`h_v` is available in [Lehnigk2021]_ and the effect of coalescence and breakup rates on the prediction of BiRD has been studied in [Hassanaly2025]_.


.. _Henry:

Multiphase flow closure models
------------

Though all closure models can be swapped by any other one available in OpenFOAM, we often use the same set of closure models. Usually, and throughout the tutorial, the user will find that the interphase drag force is obtained by using the Grace model [Grace1976]_ and transverse lift from the model by using the Tomiyama model [Tomiyama2002]_. Wall lubrication forces are computed using the model by Antal et al. [Antal1991]_ and turbulent dispersion uses the model of Burns et al. [Burns2004]_. Interphase mass transfer of species is modeled by using the mass transfer coefficient obtained from Higbie correlation [Higbie1935]_. The interphase mass transfer of a species also depends on the local saturation concentration obtained from Henry's constant and the local gas phase concentration of the species. The Higbie model is not available in OpenFOAM 9, and is described below

.. _higbie-mass-transfer-model:

Higbie Model
^^^^^^^^^^^^^^^^^^^^^^

The Higbie model calculates the phase-interface mass transfer coefficient based on the contact time of a fluid eddy at the gas-liquid boundary. The local Sherwood number (:math:`\text{Sh}`) for a spherical bubble is approximated by the analytical solution:

.. math::

   \text{Sh} = \frac{2}{\sqrt{\pi}} \sqrt{\text{Re} \cdot \text{Sc}} \approx 1.13 \sqrt{\text{Re} \cdot \text{Sc}}

where :math:`\text{Re}` is the particle Reynolds number and :math:`\text{Sc}` is the Schmidt number (which OpenFOAM computes as the product of the Lewis and Prandtl numbers, :math:`\text{Le} \cdot \text{Pr}`).

The volumetric mass transfer coefficient, :math:`K`, relates the Sherwood number to the specific interfacial area (:math:`a = 6 \alpha_g / d_b`):

In OpenFOAM's ``diffusiveMassTransferModels`` framework, the Higbie model returns a diffusivity-normalized coefficient (:math:`K`):

.. math::

   K = \frac{6 \alpha_g \text{Sh}}{d_b^2}

where :math:`\alpha_g` is the gas phase volume fraction and :math:`d_b` is the bubble diameter.

In the mass fraction transport source term :math:`K` is then multiplied by the phase density, the species diffusivity and the difference between the mass fraction at saturation (``Yf``, discussed below) and the local mass fraction.

OpenFOAM's treatment of the Henry's constant
^^^^^^^^^^^^^^^^^^^^^^

The Henry's constant is a critical parameter that controls the mass transfer rates. The Henry's constant is a temperature dependent variable that is usually expressed in :math:`mol/(kg.bar)`, see for example `the NIST database <https://webbook.nist.gov/cgi/cbook.cgi?ID=C7782447&Mask=10>`_.

In OpenFOAM, a non-dimensional version of the the Henry's constant is used and the user may find a discrepancy between the values used in the tutorials and the values available is existing databases. We will describe here how to go from one to the other.

Consider the case of a mass transfer of :math:`O_2` from the gas phase to the liquid phase. The Henry's constant in OpenFOAM is used to compute the saturated liquid mass fraction ``Yf`` of :math:`O_2` in the liquid phase as :math:`Y_l = H_{\rm CC} \frac{Y_g \rho_g}{\rho_l}` (``Foam::interfaceCompositionModels::Henry::Yf`` in ``Henry.C``), where :math:`Y_l` is ``Yf``, :math:`Y_g` is the mass fraction of :math:`O_2` in the gas, :math:`\rho_l` is the liquid density and :math:`rho_g` is the gas density, and :math:`H_{\rm CC}` is the Henry's constant as used in OpenFOAM. 

``Yf`` is eventually used to compute a mass transfer source term in ``Foam::InterfaceCompositionPhaseChangePhaseSystem<BasePhaseSystem>::
correct()`` in ``InterfaceCompositionPhaseChangePhaseSystem.C``.


Converting to non-dimensional Henry's constant
^^^^^^^^^^^^^^^^^^^^^^

:math:`H_{\rm CC}` is non-dimensional which differs from values reported in databases (noted :math:`H_{\rm CP}`). We show below how to convert :math:`H_{\rm CP}` to obtain :math:`H_{\rm CC}`.
Again, consider the case of a mass transfer of :math:`O_2` from the gas phase to the liquid phase.

 .. math::

   H_{\rm CP} = \frac{c_l'}{p' \rho_l},

where :math:`c_l'` is the concentration (:math:`mol/m^3`) and :math:`p'= x_c p` is the partial pressure of :math:`O_2`, :math:`x_c` its mole fraction in the gas phase and :math:`p` is the background pressure. 

 .. math::

   H_{\rm CC} = \frac{Y_l \rho_l}{ Y_g \rho_g},

as described above.

Using the ideal gas equation,

 .. math::

   H_{\rm CP} \rho_l R T = \frac{c_l'}{c_g'} =  \frac{y_l \rho_l}{y_g \rho_g}

where :math:`T` is the temperature, :math:`R` is the universal gas constant, and the last equation is obtained because the same molar mass applied to both the gas and the liquid phase. It comes that

 .. math::

   H_{\rm CP} \rho_l R T = H_{\rm CC}

For example, :math:`H_{\rm CP}` for :math:`O_2` is reported to be :math:`1.3 \times 10^{-8}` mol/(kg Pa) in `the NIST database <https://webbook.nist.gov/cgi/cbook.cgi?ID=C7782447&Mask=10>`_. The bubble column tutorial of BiRD uses in ``globalVars`` a value ``H_O2_298`` of 0.032, which assumed :math:`\rho_l = 1000` and :math:`T = 298`.

.. important::
   Since ``H_O2_298`` is set in ``globalVars`` it is a constant value that cannot accommodate spatial inhomogeneities of temperature
   This will be improved in future versions
   
.. _Interface tracking and compression (**OpenFOAM-13 only**):

The `birdmultiphaseEuler` module provides optional interface compression and tracking for transition from dispersed to segregated phases. The algorithm is capable of accurately track the interface and applies a compression term to keep it sharp.
This approach is still experimental and using the interface compression can lead to unphysical results and divergence.

.. _Enzymatic reactions (**OpenFOAM-13 only**):

The `birdmultiphaseEuler` module provides the `bioReactingPhaseModel` that incorporates the model of Michealis and Menten for enzymatic reactions. Currently, this model only consumes liquid species (for example, oxygen dissolved in the liquid).
For a reacting specie with mass fraction :math:`Y_l`, the uptake rate is
 
.. math::

   UR = - UR_{max} \frac{Y_i}/{k_p - \rho_l Y_i}
   
where the user parameters are the maximum uptake rate :math:`UR_{max}` and the enzyme affinity :math:`k_p`.


References
------------

.. [Lehnigk2021] Lehnigk, R. et al. (2021). "An open-source population balance modeling framework for the simulation of polydisperse multiphase flows". AIChE Journal.
.. [Hassanaly2025] Hassanaly, M. et al. (2025). "Bayesian calibration of bubble size dynamics applied to CO2 gas fermenters". Chemical Engineering Research and Design.
.. [Grace1976] Grace, J. R. (1976). "Shapes and Velocities of Single Drops and Bubbles Moving Freely through Immisicible Liquids". Transactions of the American Institute of Chemical Engineers.
.. [Tomiyama2002] Tomiyama, A. et al. (2002). "Transverse migration of single bubbles in simple shear flows". Chemical Engineering Science.
.. [Antal1991] Antal, S. P. et al. (1991). "Analysis of phase distribution in fully developed laminar bubbly two-phase flow". International Journal of Multiphase Flow.
.. [Burns2004] Burns, A. D et al. (2004). "The Favre averaged drag model for turbulent dispersion in Eulerian multi-phase flows". 5th International Conference on Multiphase Flow, ICMF.
.. [Higbie1935] Higbie, R. (1935). "The rate of absorption of pure gas into a still liquid during short periods of exposure". Transactions of the American Institute of Chemical Engineers.
.. [Menten1913] Michaelis, L., & Menten, M. L. (1913). Die Kinetik der InwertinWirkung. Biochem., (49), 333-369.






