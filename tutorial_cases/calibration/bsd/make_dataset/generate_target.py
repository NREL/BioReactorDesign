import argparse
import os

import numpy as np
from flow import Bubbles
from models import simple_nary_breakup, simple_nary_coalescence
from prettyPlot.plotting import *
from simulation import Simulation

parser = argparse.ArgumentParser(description="Target data generation")
parser.add_argument(
    "-n",
    "--nary",
    type=int,
    metavar="",
    required=False,
    help="number of bubble that coalescence and breakup per event",
    default=3,
)
parser.add_argument(
    "-br",
    "--breakup_rate",
    type=float,
    metavar="",
    required=False,
    help="Breakup rate",
    default=0.5,
)
parser.add_argument(
    "-cr",
    "--coalescence_rate",
    type=float,
    metavar="",
    required=False,
    help="Coalescence rate",
    default=0.5,
)
parser.add_argument(
    "-tf_in",
    "--input_target_file",
    type=str,
    metavar="",
    required=False,
    help="input target file to ensure that PDFs are computed for the exact same bins",
    default=None,
)
parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    help="Genrate plots",
)
args, unknown = parser.parse_known_args()

# Initialize
bubbles = Bubbles(nbubbles=2000, diam=1e-3)

dt = 0.01
nt = 15000
breakup_kwargs = {
    "breakup_rate": args.breakup_rate,
    "dt": dt,
    "n_break": args.nary,  # if n_break=2, this is binary breakup
    "min_break_diam": 1e-6,  # forbid breakup that lead to diameter < min_break_diam
}
coalescence_kwargs = {
    "coalescence_rate": args.coalescence_rate,
    "dt": dt,
    "n_coal": args.nary,  # if n_coal=2, this is binary coalescence
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

dataFolder = os.path.join("..", "data")
os.makedirs(dataFolder, exist_ok=True)

if args.input_target_file is None:
    result = sim.run(xlen=50)
else:
    result = sim.run(
        x_pdf=np.load(os.path.join(dataFolder, args.input_target_file))["x"]
    )

mean_diameter_history = result["mean_diameter_history"]
y_pdf = result["y_pdf"]
x_pdf = result["x_pdf"]
mean_bsd = np.mean(y_pdf, axis=0)
std_bsd = np.std(y_pdf, axis=0)


np.savez(
    os.path.join(
        dataFolder,
        f"target_n_{args.nary}_br_{args.breakup_rate}_cr_{args.coalescence_rate}.npz",
    ),
    x=x_pdf,
    y=mean_bsd,
    sigma=std_bsd,
)

if args.verbose:
    # Plot results
    time = np.arange(0, sim.nt * sim.dt, sim.dt)
    plt.figure(figsize=(10, 6))
    plt.plot(time, mean_diameter_history)
    pretty_labels("Time [s]", "Mean bubble diameter [m]", 14, fontname="Times")

    # Plot equilibrium BSD
    plt.figure(figsize=(10, 6))
    plt.plot(x_pdf, mean_bsd, color="k", linewidth=3)
    plt.plot(x_pdf, mean_bsd + std_bsd, "--", color="k", linewidth=3)
    plt.plot(x_pdf, mean_bsd - std_bsd, "--", color="k", linewidth=3)
    pretty_labels("diameter [m]", "Bin count", 14, fontname="Times")

    plt.show()
