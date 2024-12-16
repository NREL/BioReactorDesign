Postprocess
=====


.. _early_pred:


Perform early prediction
------------


.. code-block:: console

   python applications/early_prediction.py -df bird/postprocess/data_early

Generates

.. container:: figures-early-pred

   .. figure:: https://raw.githubusercontent.com/NREL/BioReactorDesign/main/assets/early_det.png
      :width: 70%
      :align: center
      :alt: Determinisitc early prediction

   .. figure:: https://raw.githubusercontent.com/NREL/BioReactorDesign/main/assets/early_uq.png
      :width: 70%
      :align: center
      :alt: Uncertainty-aware early prediction     


Plot conditional means
------------

.. code-block:: console

   python applications/compute_conditional_mean.py -f bird/postprocess/data_conditional_mean -avg 2

Generates (among others)

.. container:: figures-cond-mean

   .. figure:: https://raw.githubusercontent.com/NREL/BioReactorDesign/main/assets/gh_cond_mean.png
      :width: 70%
      :align: center
      :alt: Height-conditional gas holdup

   .. figure:: https://raw.githubusercontent.com/NREL/BioReactorDesign/main/assets/co2g_cond_mean.png
      :width: 70%
      :align: center
      :alt: Height-conditional CO2 gas fraction

 



