# Reproduce main results from ...

## Compare PBE simulation and Experiments

Assuming one already has installed `bird`

```bash
conda activate bird
cd validation
bash exec_comp.sh
```

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

Calibrate bubble dynamics models by also optimizing uncertainty

`bash exec_optsigma.sh`
 
