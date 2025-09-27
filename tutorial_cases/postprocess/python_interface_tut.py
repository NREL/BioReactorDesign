# Import the relevant IO functions
# Set case folder
from pathlib import Path

from bird.utilities.ofio import *

case_folder = os.path.join(
    Path(__file__).parent,
    "..",
    "..",
    "bird",
    "postprocess",
    "data_conditional_mean",
)

# Read cell centers
cell_centers, _ = read_cell_centers(case_folder)
print("cell centers shape = ", cell_centers.shape)

# Read relevant fields at time 80
co2_gas, _ = read_field(case_folder, "80", field_name="CO2.gas")
alpha_gas, _ = read_field(case_folder, "80", field_name="alpha.gas")
u_liq, _ = read_field(case_folder, "80", field_name="U.liquid")
print("cell CO2 gas shape = ", co2_gas.shape)
print("cell alpha gas shape = ", alpha_gas.shape)
print("cell u liq shape = ", u_liq.shape)

# Compute conditional average of co2_gas and alpha_gas over y
from bird.utilities.mathtools import conditional_average

y_co2_gas_cond, co2_gas_cond = conditional_average(
    cell_centers[:, 1], co2_gas, nbins=32
)
y_alpha_gas_cond, alpha_gas_cond = conditional_average(
    cell_centers[:, 1], alpha_gas, nbins=32
)

# Plot
from prettyPlot.plotting import *

fig = plt.figure()
plt.plot(y_co2_gas_cond, co2_gas_cond, color="k", label=r"$Y_{CO_2}$ [-]")
plt.plot(
    y_alpha_gas_cond, alpha_gas_cond, color="b", label=r"$\alpha_{g}$ [-]"
)
pretty_labels("Y [m]", "", fontsize=20, grid=False, fontname="Times")
pretty_legend(fontname="Times")
plt.show()

# Compute reactor quantities
from bird.postprocess.post_quantities import *

kwargs = {"case_folder": case_folder, "time_folder": "80"}
gh, field_dict = compute_gas_holdup(
    field_dict={"cell_centers": cell_centers}, **kwargs
)
print("fields stored = ", list(field_dict.keys()))
print(f"Gas Holdup = {gh:.4g}")
sup_vel, field_dict = compute_superficial_gas_velocity(
    field_dict=field_dict, **kwargs
)
print("fields stored = ", list(field_dict.keys()))
print(f"Superficial velocity = {sup_vel:.4g} m/s")
y_ave_co2, field_dict = compute_ave_y_liq(
    spec_name="CO2", field_dict=field_dict, **kwargs
)
print("fields stored = ", list(field_dict.keys()))
print(f"Reactor averaged YCO2 = {y_ave_co2:.4g}")
c_ave_co2, field_dict = compute_ave_conc_liq(
    spec_name="CO2",
    mol_weight=0.04401,
    rho_val=1000,
    field_dict=field_dict,
    **kwargs,
)
print("fields stored = ", list(field_dict.keys()))
print(f"Reactor averaged [CO2] = {c_ave_co2:.4g} mol/m3")
kla, cstar, field_dict = compute_instantaneous_kla(
    species_names=["CO2"],
    field_dict=field_dict,
    **kwargs,
)
print("fields stored = ", list(field_dict.keys()))
print(f"Reactor averaged kLa = {kla['CO2']:.4g} h-1")
print(f"Reactor averaged cstar_co2 = {cstar['CO2']:.4g} mol/m3")
diam, field_dict = compute_ave_bubble_diam(field_dict=field_dict, **kwargs)
print("fields stored = ", list(field_dict.keys()))
print(f"Reactor averaged bubble diameter = {diam:.4g} m")
