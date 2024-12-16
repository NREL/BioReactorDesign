Parameter calibration
=====

This tutorial demonstrates how to conduct the calibration with and without a surrogate forward model

Running the tutorial
------------
We assume that bird is installed within an environment called ``bird`` (see either the :ref:`Installation section for developers<installation_dev>` or the :ref:`Installation section for users<installation_users>`)

You will need to install additional dependencies

.. code-block:: console

   conda activate bird
   BIRD_DIR = `python -c "import bird; print(bird.BIRD_DIR)"`
   cd ${BIRD_DIR}/../
   pip install .[calibration]

The calibration tutorial is located at ``${BIRD_DIR}/../tutorial_cases/calibration/``

Calibration Objective
------------

The PDF of a Beta(alpha,beta) distribution is observed.
Within the context of bioreactors, one can consider this PDF to be a measured bubble size distribution (BSD)

A separate dummy numerical model is available and predicts BSD as normal distributions N(mu, sigma). The input parameter of the numerical models are mu and sigma. The objective of the modeler is to find mu and sigma such that the predicted BSD matches the observed BSD

We assume that the numerical model is expensive and instead of using the expensive numerical model for the forward model used in the calibration, one would like to use a cheaper surrogate model. Here the surrogate model is a neural network.


Building the surrogate
------------

The class ``Param_NN`` available in BiRD allows to create a parametric surrogate. Here, the input of the surrogate are the parameters ``mu`` and ``sigma``, and the variable ``x`` that parameterize the PDF.

Constructing such a surrogate is not computationally useful here since the forward simulation is cheap. This is only for illustrative purposes!

Generate dataset and train neural net surrogate with ``python tut_surrogate.py``.
The following plots should be generated. The first one shows the training and testing loss convergence. The second one shows examples of the surrogate accuracy on unseen data.


.. container:: figures-surr-train

   .. figure:: ../assets/calibration/tutorial/Loss_surr.png
      :width: 30%
      :align: center
      :alt: Loss history

   .. figure:: ../assets/calibration/tutorial/test_surr.png
      :width: 90%
      :align: center
      :alt: Test samples


Calibration with optimized likelihood uncertainty
------------

For the calibration, we can use the true function or the surrogate model. The objective PDF is assumed to be noiseless. Even though the observation is noiseless, an uncertainty is computed to account for the missing physics.
The likelihood uncertainty that represents the missing physics is optimized to ensure an uncertainty band overlap

Calibrate without a surrogate for ``alpha=5, beta=5``


.. code-block:: console

   python tut_calibration.py --alpha 5 --beta 5


.. container:: figures-cal-True-5-5

   .. figure:: ../assets/calibration/tutorial/True_opt_5.0_5.0_prop.png
      :width: 50%
      :align: center
      :alt: Calibrated prediction with the true forward model

   .. figure:: ../assets/calibration/tutorial/True_opt_5.0_5.0_corner.png
      :width: 50%
      :align: center
      :alt: Parameter PDF obtained with the true forward model


Calibrate with a surrogate for ``alpha=5, beta=5``

.. code-block:: console

   python tut_calibration.py -useNN --alpha 5 --beta 5



.. container:: figures-cal-Surr-5-5

   .. figure:: ../assets/calibration/tutorial/Surr_opt_5.0_5.0_prop.png
      :width: 50%
      :align: center
      :alt: Calibrated prediction with the surrogate forward model

   .. figure:: ../assets/calibration/tutorial/Surr_opt_5.0_5.0_corner.png
      :width: 50%
      :align: center
      :alt: Parameter PDF obtained with the surrogate forward model


Calibrate without a surrogate for ``alpha=2, beta=5``


.. code-block:: console

   python tut_calibration.py --alpha 2 --beta 5


.. container:: figures-cal-True-2-5

   .. figure:: ../assets/calibration/tutorial/True_opt_2.0_5.0_prop.png
      :width: 50%
      :align: center
      :alt: Calibrated prediction with the true forward model

   .. figure:: ../assets/calibration/tutorial/True_opt_2.0_5.0_corner.png
      :width: 50%
      :align: center
      :alt: Parameter PDF obtained with the true forward model


Calibrate with a surrogate for ``alpha=2, beta=5``

.. code-block:: console

   python tut_calibration.py -useNN --alpha 2 --beta 5



.. container:: figures-cal-True-2-5

   .. figure:: ../assets/calibration/tutorial/Surr_opt_2.0_5.0_prop.png
      :width: 50%
      :align: center
      :alt: Calibrated prediction with the surrogate forward model

   .. figure:: ../assets/calibration/tutorial/Surr_opt_2.0_5.0_corner.png
      :width: 50%
      :align: center
      :alt: Parameter PDF obtained with the surrogate forward model


Clearly, the amount of missing physics vary depending on the observations.

Using surrogate gives similar predictions as when not using a surrogate. But the surrogate was constructed with 200 forward simulations. In the case where forward simulations are expensive, the surrogate modeling approach is significantly faster.

Calibration with calibrated likelihood uncertainty
------------

The same suite can be done by calibrating the likelihood uncertainty in lieu of optimizing it (with a bissection search). This has the advantage of rapid calibration since only one calibration is needed. Here the uncertainty minimizes the negative log likelihood.

Calibrate without a surrogate for ``alpha=5, beta=5``


.. code-block:: console

   python tut_calibration.py --alpha 5 --beta 5 -cal_err


.. container:: figures-cal-calerr-True-5-5

   .. figure:: ../assets/calibration/tutorial/True_cal_5.0_5.0_prop.png
      :width: 50%
      :align: center
      :alt: Calibrated prediction with the true forward model

   .. figure:: ../assets/calibration/tutorial/True_cal_5.0_5.0_corner.png
      :width: 50%
      :align: center
      :alt: Parameter PDF obtained with the true forward model


Calibrate with a surrogate for ``alpha=5, beta=5``

.. code-block:: console

   python tut_calibration.py -useNN --alpha 5 --beta 5 -cal_err



.. container:: figures-cal-calerr-Surr-5-5

   .. figure:: ../assets/calibration/tutorial/Surr_cal_5.0_5.0_prop.png
      :width: 50%
      :align: center
      :alt: Calibrated prediction with the surrogate forward model

   .. figure:: ../assets/calibration/tutorial/Surr_cal_5.0_5.0_corner.png
      :width: 50%
      :align: center
      :alt: Parameter PDF obtained with the surrogate forward model


Calibrate without a surrogate for ``alpha=2, beta=5``


.. code-block:: console

   python tut_calibration.py --alpha 2 --beta 5 -cal_err


.. container:: figures-cal-calerr-True-2-5

   .. figure:: ../assets/calibration/tutorial/True_cal_2.0_5.0_prop.png
      :width: 50%
      :align: center
      :alt: Calibrated prediction with the true forward model

   .. figure:: ../assets/calibration/tutorial/True_cal_2.0_5.0_corner.png
      :width: 50%
      :align: center
      :alt: Parameter PDF obtained with the true forward model


Calibrate with a surrogate for ``alpha=2, beta=5``

.. code-block:: console

   python tut_calibration.py -useNN --alpha 2 --beta 5 -cal_err



.. container:: figures-cal-calerr-True-2-5

   .. figure:: ../assets/calibration/tutorial/Surr_cal_2.0_5.0_prop.png
      :width: 50%
      :align: center
      :alt: Calibrated prediction with the surrogate forward model

   .. figure:: ../assets/calibration/tutorial/Surr_cal_2.0_5.0_corner.png
      :width: 50%
      :align: center
      :alt: Parameter PDF obtained with the surrogate forward model


 



