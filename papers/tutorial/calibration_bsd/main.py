import numpy as np
from flow import Bubbles
from models import (
    simple_nary_breakup,
    simple_nary_coalescence,
    simple_normal_breakup,
)
from prettyPlot.plotting import *
from simulation import Simulation

bubbles = Bubbles(nbubbles=1000, diam=1e-4)

dt = 0.01
nt = 7000
breakup_kwargs = {
    "breakup_rate": 0.1,
    "dt": dt,
    "n_break": 2,
    "min_break_diam": 1e-6,
}
coalescence_kwargs = {
    "coalescence_rate": 0.05,
    "dt": dt,
    "n_coal": 2,
    "max_coal_diam": 1e-2,
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

mean_diam_history = sim.run()


# Plot results
time = np.arange(0, sim.nt * sim.dt, sim.dt)
plt.figure(figsize=(10, 6))
plt.plot(time, mean_diam_history)
pretty_labels("Time [s]", "Mean bubble diameter [m]", 14, fontname="Times")

# Plot equilibrium BSD
plt.figure(figsize=(10, 6))
plt.hist(sim.bubbles.diameters, bins=25)
pretty_labels("diameter [m]", "Bin count", 14, fontname="Times")

plt.show()
