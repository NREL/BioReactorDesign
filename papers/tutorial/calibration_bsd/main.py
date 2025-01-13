import numpy as np
from flow import Bubbles
from models import simple_binary_breakup, simple_binary_coalescence
from prettyPlot.plotting import *
from simulation import Simulation

bubbles = Bubbles(nbubbles=1000, diam=1e-3)

dt = 0.01
nt = 10000
breakup_kwargs = {"breakup_rate": 0.1, "dt": dt}
coalescence_kwargs = {"coalescence_rate": 0.05, "dt": dt}
sim = Simulation(
    nt=nt,
    dt=dt,
    bubbles=bubbles,
    breakup_fn=simple_binary_breakup,
    coalescence_fn=simple_binary_coalescence,
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
