bird.preprocess.dynamic\_mixer package
======================================

The ``dynamic_mixer`` package generates mixer momentum sources as OpenFOAM coded
``fvModels`` (see
:func:`~bird.preprocess.dynamic_mixer.mixing_fvModels.write_fvModel`). Two mixer
families share the same ``ball`` deposition:

* **Dynamic** (:class:`~bird.preprocess.dynamic_mixer.mixer.ActuatorMixer`, the
  ``mixers`` list) -- an *active* actuator disk driven by a power number ``Np``
  and tip speed ``Vtip``; it adds axial thrust and swirl.
* **Static** (:class:`~bird.preprocess.dynamic_mixer.mixer.StaticMixer`, the
  ``static_mixers`` list) -- a *passive* obstacle with no power input: the swirl
  is an energy-neutral axial-to-azimuthal redirection (swirl number ``S``) and
  the axial drag is a pure viscous loss (loss coefficient ``K``). It is inactive
  when the inflow opposes the mixer orientation. The momentum source follows the
  energy-neutral swirler model of Kiesewetter (2005).

Static mixer JSON schema
------------------------

Each entry of the ``static_mixers`` list accepts:

* ``S`` -- swirl number (default ``0.35``).
* ``K`` -- loss coefficient / velocity heads (default ``0.5``).
* ``radius`` -- mixer radius; a fraction of the tube in loop mode (``0.5`` spans
  the whole tube) or an absolute radius in metres in explicit mode.
* ``normal_dir`` -- mixer axis (``0``/``1``/``2`` for x/y/z).
* ``sign`` -- mixer orientation along that axis (``"+"``/``"-"``).
* ``swirl_sign`` -- rotation orientation (``"+"``/``"-"``).
* ``start_time`` -- time after which the source is active.

Placement is either explicit (``x``, ``y``, ``z``) or on a loop branch
(``branch_id`` plus ``frac_space``, the fraction along the branch), exactly as
for the dynamic mixer.

bird.preprocess.dynamic\_mixer.io\_fvModels module
--------------------------------------------------

.. automodule:: bird.preprocess.dynamic_mixer.io_fvModels
   :members:
   :undoc-members:
   :show-inheritance:

bird.preprocess.dynamic\_mixer.mixer module
-------------------------------------------

.. automodule:: bird.preprocess.dynamic_mixer.mixer
   :members:
   :undoc-members:
   :show-inheritance:

bird.preprocess.dynamic\_mixer.mixing\_fvModels module
------------------------------------------------------

.. automodule:: bird.preprocess.dynamic_mixer.mixing_fvModels
   :members:
   :undoc-members:
   :show-inheritance:

