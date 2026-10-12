### Loop reactor with actuator-disk mixers (with swirl)

Same 608 m3 loop reactor as `loop_reactor_mixing`, but the 4 mixers use the
new `ball` actuator-disk momentum source instead of the legacy `pancake` one.

The model is selected by three top-level keys in `system/mixers.json`:

- `"volumetric_source": "ball"`   — momentum deposited over a ball of radius `R`
  (a fraction of the branch cross-section), with exact momentum/torque
  conservation.
- `"power": "from_Np_Vtip"`       — the mixer power is derived from the power
  number `Np` and tip speed `Vtip` (rather than a prescribed `P`).
- `"momentum_source": "axial_and_swirl"` — adds a tangential (swirl) source on
  top of the axial thrust, set by the swirl fraction `sigma`.

Per-mixer inputs: `radius` (fraction of the branch cross-section, as for the
spargers), `Vtip` [m/s], `Np`, `sigma`, `sign` (axial push) and `swirl_sign`
(rotation sense). With `Np = 6` and `Vtip = 1.5` m/s the derived power is
~3.2 kW per mixer at this scale.

Unlike `loop_reactor_mixing`, no `constant/dynamicMix_util.H` is needed: the
Newton solve for the post-mixer velocity is inlined in the generated
`constant/fvModels`.

Single core exec

1. `bash run.sh`
