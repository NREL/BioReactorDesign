import numpy as np
from flow import Bubbles
from models import (
    simple_nary_breakup,
    simple_nary_coalescence,
    simple_normal_breakup,
)
from prettyPlot.plotting import *
from simulation import Simulation
import os

# Initialize
bubbles = Bubbles(nbubbles=2000, diam=1e-3)

dt = 0.01
nt = 15000
breakup_kwargs = {
    "breakup_rate": 0.5,
    "dt": dt,
    "n_break": 3,
    "min_break_diam": 1e-6,  # forbid breakup that lead to diameter < min_break_diam
}
coalescence_kwargs = {
    "coalescence_rate": 0.5,
    "dt": dt,
    "n_coal": 3,
    "max_coal_diam": 1e-2,  # forbid coalescence that lead to diameter > max_coal_diam
}
sim = Simulation(
    nt=nt,
    dt=dt,
    bubbles=bubbles,
    breakup_fn=simple_nary_breakup,
    coalescence_fn=simple_nary_coalescence,
    breakup_kwargs=breakup_kwargs,
    coalescence_kwargs=coalescence_kwargs,
)


result = sim.run(xlen=50)
mean_diameter_history = result["mean_diameter_history"]
y_pdf = result["y_pdf"]
x_pdf = result["x_pdf"]

# Plot results
time = np.arange(0, sim.nt * sim.dt, sim.dt)
plt.figure(figsize=(10, 6))
plt.plot(time, mean_diameter_history)
pretty_labels("Time [s]", "Mean bubble diameter [m]", 14, fontname="Times")

# Plot equilibrium BSD
plt.figure(figsize=(10, 6))
mean_bsd = np.mean(y_pdf, axis=0)
std_bsd = np.std(y_pdf, axis=0)
plt.plot(x_pdf, mean_bsd, color="k", linewidth=3)
plt.plot(x_pdf, mean_bsd+std_bsd, "--", color="k", linewidth=3)
plt.plot(x_pdf, mean_bsd-std_bsd, "--", color="k", linewidth=3)
pretty_labels("diameter [m]", "Bin count", 14, fontname="Times")

dataFolder = os.path.join("..","data")
os.makedirs(dataFolder, exist_ok=True)

np.savez(os.path.join(dataFolder,"target.npz"), x=x_pdf, y=mean_bsd, sigma=std_bsd)

plt.show()
