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
import argparse

parser = argparse.ArgumentParser(description="Generate data for training the surrogate")
parser.add_argument(
    "-filesave",
    "--filesave",
    type=str,
    metavar="",
    required=False,
    help="dataset file",
    default="dataset.npz",
)
args, unknown = parser.parse_known_args()
data_folder = os.path.join("..","data")
target = np.load(os.path.join(data_folder,"target.npz"))
x_pdf_target = target["x"]
beff_fact_data = []
ceff_data = []
x_pdf_data = [] 
y_pdf_data = [] 

try:
    os.remove(args.filesave) 
except FileNotFoundError:
    pass

dt = 0.01
nt = 15000
isim=0
nsim=100
while isim<nsim:
    isim += 1
    ceff = np.random.uniform(0.02, 2)
    beff_fact = np.random.uniform(0.8, 1.1)
    
    # Initialize
    bubbles = Bubbles(nbubbles=2000, diam=1e-3)
    breakup_kwargs = {
        "breakup_rate": ceff*beff_fact,
        "dt": dt,
        "n_break": 2,  # if n_break=2, this is binary breakup
        "min_break_diam": 1e-6,  # forbid breakup that lead to diameter < min_break_diam
    }
    coalescence_kwargs = {
        "coalescence_rate": ceff,
        "dt": dt,
        "n_coal": 2,  # if n_coal=2, this is binary coalescence
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
    
    result = sim.run(window_size=200,kill_failed_sim=True, x_pdf=x_pdf_target)
    if "x_pdf" in result:
        mean_diameter_history = result["mean_diameter_history"]
        y_pdf = result["y_pdf"]
        x_pdf = result["x_pdf"]
        start_ave_time = result["start_ave_time"]           
    
        # Plot results
        time = np.arange(0, sim.nt * sim.dt, sim.dt)
        plt.figure(figsize=(10, 6))
        plt.plot(time, mean_diameter_history)
        plt.plot(np.ones(10)*start_ave_time, np.linspace(np.amin(mean_diameter_history), np.amax(mean_diameter_history), 10), '--', color='k')
        pretty_labels("Time [s]", "Mean bubble diameter [m]", 14, fontname="Times")
        plt.close()
        
        # Plot equilibrium BSD
        plt.figure(figsize=(10, 6))
        mean_bsd = np.mean(y_pdf, axis=0)
        std_bsd = np.std(y_pdf, axis=0)
        plt.plot(x_pdf, mean_bsd, color="k", linewidth=3)
        plt.plot(x_pdf, mean_bsd+std_bsd, "--", color="k", linewidth=3)
        plt.plot(x_pdf, mean_bsd-std_bsd, "--", color="k", linewidth=3)
        pretty_labels("diameter [m]", "Bin count", 14, fontname="Times")
        plt.close()
        #plt.show()
        print(f"\nSUCCESS beff_fact = {beff_fact} ceff = {ceff}\n")
        beff_fact_data.append(beff_fact)
        ceff_data.append(ceff)
        y_pdf_data.append(mean_bsd)
        np.savez(os.path.join(data_folder,args.filesave), x=x_pdf_target, y=np.array(y_pdf_data), beff_fact=np.array(beff_fact_data), ceff=np.array(ceff_data))
    else:
        print(f"\nFAILED beff_fact = {beff_fact} ceff = {ceff}\n")
        beff_fact_data.append(beff_fact)
        ceff_data.append(ceff)
        y_pdf_data.append(np.ones(len(x_pdf_target)))
        np.savez(os.path.join(data_folder,args.filesave), x=x_pdf_target, y=np.array(y_pdf_data), beff_fact=np.array(beff_fact_data), ceff=np.array(ceff_data))
    

