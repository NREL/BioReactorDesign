# Reproduce main results from ...

## Compare PBE simulation and Experiments

Assuming one already has installed `bird`

```bash
conda activate bird
cd validation
bash exec_comp.sh
```
Generates these figures in the `Figures/` folder

Compares simulation to experiments

<p float="left">
  <img src="https://github.com/NREL/BioReactorDesign/tree/main/assets/validation/co2.png" width="350"/>
  <img src="https://github.com/NREL/BioReactorDesign/tree/main/assets/validation/gh.png" width="350"/>
</p>


Checks that simulations are converged based on the gas holdup history

<p float="center">
  <img src="https://github.com/NREL/BioReactorDesign/tree/main/assets/validation/conv.png" width="350"/>
</p>


## Calibrate models using experiments

Assuming one already has installed `bird`
Extra packages are needed

```bash
conda activate bird
pip install -r requirements.txt
cd calibration
```

Examples are shown for binary breakup model but also apply to the breakup model.

`cd binBreakup`

Calibrate bubble dynamics models by also calibrating uncertainty

`bash exec_calsigma.sh`

Generates this corner plot in the `Figures/` folder. Left is for binary breakup and right is for breakup.

<p float="left">
  <img src="https://github.com/NREL/BioReactorDesign/tree/main/assets/calibration/corner_bb_calsigm.png" width="350"/>
  <img src="https://github.com/NREL/BioReactorDesign/tree/main/assets/calibration/corner_b_calsigm.png" width="350"/>
</p>

Calibrate bubble dynamics models by also optimizing uncertainty. This script will take 10 times longer than the previous script as it repeats the calculations for different uncertainty levels.

`bash exec_optsigma.sh`

Generates this corner plot in the `Figures/` folder. Left is for binary breakup and right is for breakup.
 
<p float="left">
  <img src="https://github.com/NREL/BioReactorDesign/tree/main/assets/calibration/corner_bb_optsigm.png" width="350"/>
  <img src="https://github.com/NREL/BioReactorDesign/tree/main/assets/calibration/corner_b_optsigm.png" width="350"/>
</p>
