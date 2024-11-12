# Calibration tutorial

This tutorial demonstrates how to conduct the calibration with and without a surrogate forward model

## Running the tutorial
We assume that bird is installed within an environment called `bird` (see installation instruction at the root of the repo)
You will need to install additional dependencies
```
conda activate bird
cd ${BIRD_DIR}/../papers/tutorial/calibration
pip install -r requirememts.txt
```
Note that `${BIRD_DIR}` is not an environment variable, and should be replaced by the path to the root of the repo.

## Calibration Objective

The PDF of a Beta(alpha,beta) distribution is observed.
Within the context of bioreactors, one can consider this PDF to be a measured bubble size distribution (BSD)

A separate dummy numerical model is available and predicts BSD as normal distributions N(mu, sigma). The input parameter of the numerical models are mu and sigma. The objective of the modeler is to find mu and sigma such that the predicted BSD matches the observed BSD

We assume that the numerical model is expensive and instead of using the expensive numerical model for the forward model used in the calibration, one would like to use a cheaper surrogate model. Here the surrogate model is a neural network.


## Building the surrogate

The class `Param_NN` available in BiRD allows to create a parametric surrogate. Here, the input of the surrogate are the parameters `mu` and `sigma`, and the variable `x` that parameterize the PDF.

Constructing such a surrogate is not computationally useful here since the forward simulation is cheap. This is only for illustrative purposes!

Generate dataset and train neural net surrogate with `python tut_surrogate.py` 

<p align="center">
<img src="/papers/tutorial/calibration/assets/Loss_surr.png" width="150" height="125"/>
<img src="/papers/tutorial/calibration/assets/test_surr.png" width="625" height="125"/>
</p>


## Calibration

For the calibration, we can use the true function or the surrogate model. The objective PDF is assumed to be noiseless. Even though the observation is noiseless, an uncertainty is computed to account for the missing physics.

Calibrate without a surrogate for `alpha=5, beta=5`: `python tut_calibration_all.py --alpha 5 --beta 5`

<p align="center">
<img src="/papers/tutorial/calibration/assets/True_a_5_b_5_prop.png" width="150" height="125"/>
<img src="/papers/tutorial/calibration/assets/True_a_5_b_5_corner.png" width="150" height="125"/>
</p>


Calibrate without a surrogate for `alpha=2, beta=5`: `python tut_calibration_all.py --alpha 2 --beta 5`


<p align="center">
<img src="/papers/tutorial/calibration/assets/True_a_2_b_5_prop.png" width="150" height="125"/>
<img src="/papers/tutorial/calibration/assets/True_a_2_b_5_corner.png" width="150" height="125"/>
</p>

Clearly, the amount of missing physics vary depending on the observations.

Then the same exercise can be done when using a neural net surrogate

Calibrate with a surrogate for `alpha=5, beta=5`: `python tut_calibration_all.py --useNN --alpha 5 --beta 5`

<p align="center">
<img src="/papers/tutorial/calibration/assets/Surr_a_5_b_5_prop.png" width="150" height="125"/>
<img src="/papers/tutorial/calibration/assets/Surr_a_5_b_5_corner.png" width="150" height="125"/>
</p>


Calibrate without a surrogate for `alpha=2, beta=5`: `python tut_calibration_all.py --useNN --alpha 2 --beta 5`


<p align="center">
<img src="/papers/tutorial/calibration/assets/Surr_a_2_b_5_prop.png" width="150" height="125"/>
<img src="/papers/tutorial/calibration/assets/Surr_a_2_b_5_corner.png" width="150" height="125"/>
</p>

Using surrogate gives similar predictions as when not using a surrogate. But the surrogate was constructed with 200 forward simulations.
