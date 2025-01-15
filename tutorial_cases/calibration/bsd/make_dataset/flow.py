import numpy as np
from prettyPlot.plotting import *
from utils import check_vol_cons, diam2vol, vol2diam


class Bubbles:
    def __init__(
        self,
        diameters: np.ndarray | None = None,
        nbubbles: int | None = None,
        diam: float | None = None,
    ):
        if diameters is not None:
            self.diameters = diameters
        else:
            assert nbubbles is not None
            assert diam is not None
            self.diameters = np.ones(nbubbles) * diam
        self.mean_diameter = None
        self.update_mean()
        self.breakup_rate_factor = 1.0
        self.coalescence_rate_factor = 1.0

    def update_mean(self):
        self.mean_diameter = np.mean(self.diameters)

    def coalesce(self, id_in: list[int], new_diameter: float = None) -> None:

        check_vol_cons(
            diam_in=[self.diameters[id_b] for id_b in id_in],
            diam_out=[new_diameter],
        )

        # Update bubble diameters
        self.diameters = np.delete(self.diameters, [id_in])
        self.diameters = np.append(self.diameters, new_diameter)
        # Reset the coalescence rate factor
        # This factor is useful for very small timesteps (this is not the efficiency factor)
        self.coalescence_rate_factor = 1.0

    def breakup(self, id_break: int, new_diam_list: list[float]) -> None:
        check_vol_cons([self.diameters[id_break]], new_diam_list)
        # Update bubble diameters
        self.diameters = np.delete(self.diameters, id_break)
        self.diameters = np.append(self.diameters, new_diam_list)
        # Reset the breakup rate factor
        # This factor is useful for very small timesteps (this is not the efficiency factor)
        self.breakup_rate_factor = 1.0
