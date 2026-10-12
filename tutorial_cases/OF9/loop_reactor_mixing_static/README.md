### Loop reactor with passive static mixers

Same 608 m3 loop reactor as `loop_reactor_mixing`, but the mixers are **passive
static mixers** instead of the powered actuator disks of
`loop_reactor_mixing_swirl`.

The static mixer injects no power: the swirl is imposed as an energy-neutral
axial-to-azimuthal redirection (Kiesewetter's swirler model) and the axial drag
is a pure viscous loss. It is selected simply by providing a `static_mixers`
list in `system/mixers.json` (the `ball` deposition is triggered automatically),
with per-mixer inputs:

- `S`          — swirl number (target ratio of azimuthal to axial momentum flux).
- `K`          — loss coefficient (velocity heads) for the axial pressure drop.
- `radius`     — mixer radius as a fraction of the branch cross-section (`0.5`
  spans the whole tube), as for the spargers.
- `sign`       — mixer orientation; the source is inactive when the inflow
  opposes it.
- `swirl_sign` — rotation orientation.
- `start_time` — time after which the source is active.

Placement uses `branch_id` + `frac_space`, as for the dynamic mixer. This case
places one `+`-oriented and one `-`-oriented mixer so both sign paths are
exercised.

Single core exec

1. `bash run.sh`
