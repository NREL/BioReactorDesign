import numpy as np
from flow import Bubbles
from prettyPlot.progressBar import print_progress_bar
from utils import check_conv, get_bsd

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

    def init_pdf(self, xlen:int=100) -> np.ndarray:
        mindiam = 0
        maxdiam = np.amax(self.bubbles.diameters) * 1.5
        x_pdf = np.linspace(mindiam, maxdiam, xlen)
        return x_pdf

    def run(self, window_size:int=100, mean_tol:float=1e-8, std_tol:float=np.inf, kill_failed_sim:bool=False, x_pdf:None|np.ndarray=None, xlen:int=100) -> dict:
        print_progress_bar(
            0,
            self.nt,
            prefix=f"No. bubbles = {len(self.bubbles.diameters)}, mean diam = {self.bubbles.mean_diameter}, Step 0 / {self.nt} ",
            suffix="Complete",
            length=50,
        )
        # Time evolution
        mean_diameter_history = []
        y_pdf = []
        start_ave = False
        init_no_bubbles = len(self.bubbles.diameters)
        for i_t in range(1, self.nt + 1):
            self.time = i_t * self.dt
            self.breakup_fn(self.bubbles, self.breakup_kwargs)
            self.coalescence_fn(self.bubbles, self.coalescence_kwargs)
            self.bubbles.update_mean()
            mean_diameter_history.append(self.bubbles.mean_diameter)
            n_bubbles = len(self.bubbles.diameters)
            # If number of bubbles is growing to infinity or decreasing to 0, stop the run
            if kill_failed_sim and (n_bubbles>(init_no_bubbles*10) or n_bubbles<(init_no_bubbles*0.1)):
                return {}
            # Check when we are in statistically stationary region
            if not start_ave:
                converged = check_conv(mean_diameter_history, window_size, mean_tol, std_tol)
                if converged:
                    start_ave = True
                    start_ave_time = self.time
                    if x_pdf is None:
                        x_pdf = self.init_pdf(xlen)
                    # If convergence has not happened by the end the simulation, return failure
                    if start_ave_time>=self.total_time:
                        return {}
            # If we are in statistically stationary state, start averaging the BSD
            if start_ave:
                y_pdf.append(get_bsd(x_pdf, self.bubbles.diameters))
          

            print_progress_bar(
                i_t,
                self.nt,
                prefix=f"No. bubbles = {n_bubbles}, mean diam = {self.bubbles.mean_diameter:.2g}, Step {i_t+1} / {self.nt} ",
                suffix="Complete",
                length=50,
            )
        # If convergence has not happened return failure
        if not start_ave:
            return {}
        result_dict = {"mean_diameter_history": np.array(mean_diameter_history), "y_pdf": np.array(y_pdf), "x_pdf": np.array(x_pdf), 'start_ave_time': start_ave_time} 
        return result_dict
