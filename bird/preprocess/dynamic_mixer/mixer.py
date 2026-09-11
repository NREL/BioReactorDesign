import math

from bird import logger


def actuator_disk_power(
    Np: float, Vtip: float, R: float, rho: float = 1000.0
) -> float:
    """Power [W] drawn by a ``from_Np_Vtip`` actuator-disk mixer.

    ``P = Np * rho * Vtip**3 * D**2 / pi**3`` with ``D = 2R`` (see the ball
    mixer derivation). Used at case-generation time to record the derived
    power for the efficiency QoI.
    """
    D = 2.0 * R
    return Np * rho * Vtip**3 * D**2 / math.pi**3


class Mixer:
    def __init__(self):
        self.x = None
        self.y = None
        self.z = None
        self.rad = 0.015
        self.power = 300
        self.start_time = 1.0
        self.smear = 3
        self.sign = None
        self.normal_dir = None
        self.ready = False

    def update_from_expl_dict(self, mixer_dict):
        if "x" in mixer_dict:
            self.x = mixer_dict["x"]
        if "y" in mixer_dict:
            self.y = mixer_dict["y"]
        if "z" in mixer_dict:
            self.z = mixer_dict["z"]
        if "rad" in mixer_dict:
            self.rad = mixer_dict["rad"]
        if "power" in mixer_dict:
            self.power = mixer_dict["power"]
        if "sign" in mixer_dict:
            self.sign = mixer_dict["sign"]
        if "smear" in mixer_dict:
            self.smear = mixer_dict["smear"]
        if "start_time" in mixer_dict:
            self.start_time = mixer_dict["start_time"]
        if "normal_dir" in mixer_dict:
            self.normal_dir = mixer_dict["normal_dir"]
        self.check_status()

    def update_from_loop_dict(self, mixer_dict, geom_dict, mesh_dict=None):
        segment = geom_dict["segments"][mixer_dict["branch_id"]]
        pos = segment["start"] + mixer_dict["frac_space"] * segment["conn"]
        self.x = pos[0]
        self.y = pos[1]
        self.z = pos[2]
        self.normal_dir = segment["normal_dir"]
        if "rad" in mixer_dict:
            self.rad = min(mixer_dict["rad"], segment["max_rad"])
        else:
            self.rad = segment["max_rad"] * 0.7
        if "power" in mixer_dict:
            self.power = mixer_dict["power"]
        if "sign" in mixer_dict:
            self.sign = mixer_dict["sign"]
        if "start_time" in mixer_dict:
            self.start_time = mixer_dict["start_time"]
        if "normal_dir" in mixer_dict:
            self.normal_dir = mixer_dict["normal_dir"]
        if mesh_dict is not None:
            if self.normal_dir == 0:
                min_mesh_transv = min(
                    mesh_dict["Blockwise"]["y"], mesh_dict["Blockwise"]["z"]
                )
            elif self.normal_dir == 1:
                min_mesh_transv = min(
                    mesh_dict["Blockwise"]["x"], mesh_dict["Blockwise"]["z"]
                )
            elif self.normal_dir == 2:
                min_mesh_transv = min(
                    mesh_dict["Blockwise"]["x"], mesh_dict["Blockwise"]["y"]
                )
            self.smear = min_mesh_transv // 3
        self.check_status(blocks=segment["blocks"])

    def check_status(self, blocks=None):
        if (
            self.x is None
            or self.y is None
            or self.z is None
            or self.normal_dir is None
            or ((not self.sign == "+") and (not self.sign == "-"))
        ):
            self.ready = False
        else:
            logger.info(
                f"\n\tpos({self.x:.2g}, {self.y:.2g}, {self.z:.2g})"
                + f"\n\tnormal_dir {self.normal_dir}"
                + f"\n\trad {self.rad:.2g}"
                + f"\n\tpower {self.power:.2g}"
                + f"\n\tsign {self.sign}"
                + f"\n\tsmear {self.smear}"
                + f"\n\tstart_time {self.start_time:.2g}"
            )
            if blocks is not None:
                logger.info(f"\tbranch = {blocks}")

            self.ready = True


class ActuatorMixer:
    """Actuator-disk mixer with optional swirl, used by the ``ball`` source.

    Unlike :class:`Mixer` (the legacy ``pancake`` source), momentum is deposited
    over a ball of physical radius ``R`` and the drive is set by a power number
    ``Np`` and tip speed ``Vtip`` rather than a raw power. ``sigma`` is the swirl
    fraction; ``sign`` is the axial push direction and ``swirl_sign`` the
    (independent) rotation sense.
    """

    def __init__(self):
        self.x = None
        self.y = None
        self.z = None
        self.normal_dir = None
        self.R = None  # physical mixer radius [m]
        self.Vtip = 1.5  # tip speed [m/s]
        self.Np = 6.0  # power number [-]
        self.sigma = 0.35  # swirl fraction [-]
        self.power = None  # mixer power P [W], only used when power=from_P
        self.sign = None  # axial push sign, "+" / "-"
        self.swirl_sign = "+"  # rotation sense, "+" / "-"
        self.start_time = 1.0
        self.ready = False

    def _read_common(self, mixer_dict: dict) -> None:
        """Read the per-mixer keys"""
        if "Vtip" in mixer_dict:
            self.Vtip = mixer_dict["Vtip"]
        if "Np" in mixer_dict:
            self.Np = mixer_dict["Np"]
        if "sigma" in mixer_dict:
            self.sigma = mixer_dict["sigma"]
        if "power" in mixer_dict:
            self.power = mixer_dict["power"]
        if "sign" in mixer_dict:
            self.sign = mixer_dict["sign"]
        if "swirl_sign" in mixer_dict:
            self.swirl_sign = mixer_dict["swirl_sign"]
        if "start_time" in mixer_dict:
            self.start_time = mixer_dict["start_time"]

    def update_from_expl_dict(self, mixer_dict: dict) -> None:
        """Populate from an explicit mixer dict (absolute position and radius)."""
        if "x" in mixer_dict:
            self.x = mixer_dict["x"]
        if "y" in mixer_dict:
            self.y = mixer_dict["y"]
        if "z" in mixer_dict:
            self.z = mixer_dict["z"]
        if "normal_dir" in mixer_dict:
            self.normal_dir = mixer_dict["normal_dir"]
        if "radius" in mixer_dict:
            # explicit mode: radius is absolute [m]
            self.R = mixer_dict["radius"]
        self._read_common(mixer_dict)
        self.check_status()

    def update_from_loop_dict(self, mixer_dict: dict, geom_dict: dict) -> None:
        """Populate from a loop mixer dict.

        Parameters
        ----------
        mixer_dict: dict
            Mixer entry with ``branch_id``, ``frac_space`` and, optionally,
            ``radius`` as a fraction of the branch cross-section.
        geom_dict: dict
            Output of ``from_block_rect_to_seg`` (``segments`` and ``blocksize``).
        """
        segment = geom_dict["segments"][mixer_dict["branch_id"]]
        pos = segment["start"] + mixer_dict["frac_space"] * segment["conn"]
        self.x = float(pos[0])
        self.y = float(pos[1])
        self.z = float(pos[2])
        self.normal_dir = segment["normal_dir"]
        # radius is a fraction of the branch cross-section (as for spargers):
        # R = frac * mean of the two block sizes transverse to the axis.
        bx, by, bz = geom_dict["blocksize"]
        transverse = {0: (by, bz), 1: (bx, bz), 2: (bx, by)}[self.normal_dir]
        frac = mixer_dict.get("radius", 0.4)
        self.R = frac * 0.5 * (transverse[0] + transverse[1])
        self._read_common(mixer_dict)
        self.check_status(blocks=segment["blocks"])

    def check_status(self, blocks=None) -> None:
        """Log the resolved mixer and set ``ready`` if all fields are present."""
        if (
            self.x is None
            or self.y is None
            or self.z is None
            or self.normal_dir is None
            or self.R is None
            or self.sign not in ("+", "-")
            or self.swirl_sign not in ("+", "-")
        ):
            self.ready = False
        else:
            logger.info(
                f"\n\tpos({self.x:.2g}, {self.y:.2g}, {self.z:.2g})"
                + f"\n\tnormal_dir {self.normal_dir}"
                + f"\n\tR {self.R:.2g}"
                + f"\n\tVtip {self.Vtip:.2g}"
                + f"\n\tNp {self.Np:.2g}"
                + f"\n\tsigma {self.sigma:.2g}"
                + f"\n\tsign {self.sign}  swirl_sign {self.swirl_sign}"
                + f"\n\tstart_time {self.start_time:.2g}"
            )
            if blocks is not None:
                logger.info(f"\tbranch = {blocks}")
            self.ready = True


class StaticMixer:
    """Passive (unpowered) momentum-source mixer used by the ``ball`` source.

    Unlike :class:`ActuatorMixer`, no power is injected: the swirl is imposed as
    an energy-neutral axial-to-azimuthal redirection and the axial drag is a pure
    loss. The drive is a swirl number ``S`` and a loss coefficient ``K`` rather
    than ``Np``/``Vtip``; ``sign`` is the mixer orientation (the source is
    inactive when the inflow opposes it) and ``swirl_sign`` the rotation sense.
    """

    def __init__(self):
        self.x = None
        self.y = None
        self.z = None
        self.normal_dir = None
        self.R = None  # physical mixer radius [m]
        self.S = 0.35  # swirl number [-]
        self.K = 0.5  # loss coefficient [-]
        self.sign = None  # mixer orientation, "+" / "-"
        self.swirl_sign = "+"  # rotation sense, "+" / "-"
        self.start_time = 1.0
        self.ready = False

    def _read_common(self, mixer_dict: dict) -> None:
        """Read the per-mixer keys."""
        if "S" in mixer_dict:
            self.S = mixer_dict["S"]
        if "K" in mixer_dict:
            self.K = mixer_dict["K"]
        if "sign" in mixer_dict:
            self.sign = mixer_dict["sign"]
        if "swirl_sign" in mixer_dict:
            self.swirl_sign = mixer_dict["swirl_sign"]
        if "start_time" in mixer_dict:
            self.start_time = mixer_dict["start_time"]

    def update_from_expl_dict(self, mixer_dict: dict) -> None:
        """Populate from an explicit mixer dict (absolute position and radius)."""
        if "x" in mixer_dict:
            self.x = mixer_dict["x"]
        if "y" in mixer_dict:
            self.y = mixer_dict["y"]
        if "z" in mixer_dict:
            self.z = mixer_dict["z"]
        if "normal_dir" in mixer_dict:
            self.normal_dir = mixer_dict["normal_dir"]
        if "radius" in mixer_dict:
            # explicit mode: radius is absolute [m]
            self.R = mixer_dict["radius"]
        self._read_common(mixer_dict)
        self.check_status()

    def update_from_loop_dict(self, mixer_dict: dict, geom_dict: dict) -> None:
        """Populate from a loop mixer dict.

        Parameters
        ----------
        mixer_dict: dict
            Mixer entry with ``branch_id``, ``frac_space`` and, optionally,
            ``radius`` as a fraction of the branch cross-section (``0.5`` spans
            the whole tube).
        geom_dict: dict
            Output of ``from_block_rect_to_seg`` (``segments`` and ``blocksize``).
        """
        segment = geom_dict["segments"][mixer_dict["branch_id"]]
        pos = segment["start"] + mixer_dict["frac_space"] * segment["conn"]
        self.x = float(pos[0])
        self.y = float(pos[1])
        self.z = float(pos[2])
        self.normal_dir = segment["normal_dir"]
        # radius is a fraction of the branch cross-section (as for the actuator
        # mixer): R = frac * mean of the two block sizes transverse to the axis.
        bx, by, bz = geom_dict["blocksize"]
        transverse = {0: (by, bz), 1: (bx, bz), 2: (bx, by)}[self.normal_dir]
        frac = mixer_dict.get("radius", 0.4)
        self.R = frac * 0.5 * (transverse[0] + transverse[1])
        self._read_common(mixer_dict)
        self.check_status(blocks=segment["blocks"])

    def check_status(self, blocks=None) -> None:
        """Log the resolved mixer and set ``ready`` if all fields are present."""
        if (
            self.x is None
            or self.y is None
            or self.z is None
            or self.normal_dir is None
            or self.R is None
            or self.sign not in ("+", "-")
            or self.swirl_sign not in ("+", "-")
        ):
            self.ready = False
        else:
            logger.info(
                f"\n\tpos({self.x:.2g}, {self.y:.2g}, {self.z:.2g})"
                + f"\n\tnormal_dir {self.normal_dir}"
                + f"\n\tR {self.R:.2g}"
                + f"\n\tS {self.S:.2g}"
                + f"\n\tK {self.K:.2g}"
                + f"\n\tsign {self.sign}  swirl_sign {self.swirl_sign}"
                + f"\n\tstart_time {self.start_time:.2g}"
            )
            if blocks is not None:
                logger.info(f"\tbranch = {blocks}")
            self.ready = True
