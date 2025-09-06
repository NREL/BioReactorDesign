Uncertainty quantification
=====


.. _early_pred:


Perform early prediction
------------


.. code-block:: console

   python applications/early_prediction.py -df bird/postprocess/data_early

Generates

.. container:: figures-early-pred

   .. figure:: ../assets/early_det.png
      :width: 70%
      :align: center
      :alt: Determinisitc early prediction

   .. figure:: ../assets/early_uq.png
      :width: 70%
      :align: center
      :alt: Uncertainty-aware early prediction     


Compute kLa with uncertainty estimates
------------

Based on the time-history of the concentration of a species, one can calculate kLa by fitting the function

.. math::

   (C^* - C_0) (1-\exp(-kLa (t-t_0)) + C_0 

where :math:`C^*` is the equilibrium concentration (to be fitted), :math:`C_0` is the initial concentration, :math:`t` is time, :math:`t_0` is the initial time after which concentration is recorded 

Accurate estimates can be obtained if sufficient data is acquired. Otherwise, it may be useful to derive uncertainty estimates about :math:`C^*` and :math:`kLa` (the parameters fitted)

This can be achieved with a Bayesian calibration procedure. The calibration is conducted by removing transient data, and by doing a data bootstrapping. The number of data to remove in the transient phase is automatically determined by examining how accurate is the fit.


.. code-block:: console

   python applications/compute_kla_uq.py -i bird/postprocess/data_kla/volume_avg.dat -ti 0 -ci 1 -mc 10 

Generates

.. code-block:: console

   Chopping index = 0
   Chopping index = 1
   Chopping index = 2
   Chopping index = 3
   Chopping index = 4
   Doing data bootstrapping
   	 scenario 0
   	 scenario 1
   	 scenario 2
   	 scenario 3
   For bird/postprocess/data_kla/volume_avg.dat with time index: 0, concentration index: 1
   	kla = 0.09005 +/- 0.0006387
   	cstar = 0.3107 +/- 0.0006122
   Without data bootstrap
   	kla = 0.09014 +/- 0.0005957
   	cstar = 0.3105 +/- 0.0005472


Compute mean statistics with uncertainty
------------

Averaging a discretized time-series signal is used in many contexts to characterize bio reactors (to compute averaged holdup or species concentrations). Averaging is subject to statistical error and we provide tools to manage it. 

The run the illustrative example we consider here:
First enable ``setup_logging(level="DEBUG")`` in ``bird/__init__.py``.
Second, run

.. code-block:: console

   python applications/compute_time_series_mean.py

There, we consider a time series acquired over the interval :math:`[0, 2]` where the signal is :math:`cos (2 \pi t)` shown below 

.. figure:: ../assets/time_series.png
      :width: 70%
      :align: center
      :alt: Time series example

We can sample the signal with 100 points through the interval :math:`[0, 2]`, and we obtain the following output

.. code-block:: console

   2025-09-02 12:36:15,016 [DEBUG] bird: Making the time series equally spaced over time
   2025-09-02 12:36:15,016 [DEBUG] bird: Time series already equally spaced
   2025-09-02 12:36:15,016 [DEBUG] bird: T0 = 1.270553916086648
   2025-09-02 12:36:15,016 [INFO] bird: Mean = 0.01 +/- 0.081

The ``T0`` value suggests that every :math:`1.27` points is considered independent. The uncertainty about the mean is estimated via the central limit theorem, where the number of datapoints is downsampled to make the sample independents

We can also oversample the signal with 100 times more points. No more information has been provided about the signal, but without identifying the number of steps over which samples can be considered independent, the uncertainty (:math:`0.081`) would be artificially reduced to :math:`0.0081`.

Here we obtain 

.. code-block:: console

   2025-09-02 12:36:15,016 [DEBUG] bird: Making the time series equally spaced over time
   2025-09-02 12:36:15,016 [DEBUG] bird: Time series already equally spaced
   2025-09-02 12:36:15,030 [DEBUG] bird: T0 = 126.6515206796017
   2025-09-02 12:36:15,030 [INFO] bird: Mean = 0.0001 +/- 0.08

The mean calculation function identifies that every 127 points is independent, and the uncertainty about the mean is not artificially reduced.
