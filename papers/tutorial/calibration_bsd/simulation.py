import numpy as np
from flow import Bubbles
from prettyPlot.progressBar import print_progress_bar

class Simulation:
    def __init__(
        self,
        nt: int,
        dt: float,
        bubbles: Bubbles,
        breakup_fn,
        coalescence_fn,
        breakup_kwargs,
        coalescence_kwargs,
    ):
        self.nt = nt
        self.dt = dt
        self.time = 0
        self.total_time = nt * dt
        self.bubbles = bubbles
        self.breakup_fn = breakup_fn
        self.coalescence_fn = coalescence_fn
        self.breakup_kwargs = breakup_kwargs
        self.coalescence_kwargs = coalescence_kwargs

    def run(self) -> list[float]:
        print_progress_bar(
            0,  
            self.nt,
            prefix=f"No. bubbles = {len(self.bubbles.diameters)}, mean diam {self.bubbles.mean_diameter}, Step 0 / {self.nt} ",
            suffix="Complete",
            length=50,
        )  
        # Time evolution
        mean_diameter_history = []
        for i_t in range(1, self.nt + 1):
            self.time = i_t * self.dt
            self.breakup_fn(self.bubbles, self.breakup_kwargs)
            self.coalescence_fn(self.bubbles, self.coalescence_kwargs)
            # breakpoint()
            self.bubbles.update_mean()
            mean_diameter_history.append(self.bubbles.mean_diameter)
            #print(len(self.bubbles.diameters))
            #print(np.amin(self.bubbles.diameters))
            print_progress_bar(
                i_t,  
                self.nt,
                prefix=f"No. bubbles = {len(self.bubbles.diameters)}, mean diam {self.bubbles.mean_diameter:.2g}, Step {i_t+1} / {self.nt} ",
                suffix="Complete",
                length=50,
            )  
        return mean_diameter_history
