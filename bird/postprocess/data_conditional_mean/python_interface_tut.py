# Import the relevant IO functions
from bird.utilities.ofio import *

# Read cell centers
cell_centers, _ = read_cell_centers(".")
print("cell centers shape = ", cell_centers.shape)

# Read relevant fields at time 80
co2_gas = readOF("80/CO2.gas")
alpha_gas = readOF("80/alpha.gas")
u_liq = readOF("80/U.liquid")
print("cell CO2 gas shape = ", co2_gas["field"].shape)
print("cell alpha gas shape = ", alpha_gas["field"].shape)
print("cell u liq shape = ", u_liq["field"].shape)

# Compute conditional average of co2_gas and alpha_gas over y
from bird.utilities.mathtools import conditional_average

y_co2_gas_cond, co2_gas_cond = conditional_average(
    cell_centers[:, 1], co2_gas["field"], nbins=32
)
y_alpha_gas_cond, alpha_gas_cond = conditional_average(
    cell_centers[:, 1], alpha_gas["field"], nbins=32
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

kwargs = {"case_folder": ".", "time_folder": "80"}
gh, field_dict = compute_gas_holdup(
    volume_time="1", field_dict={"cell_centers": cell_centers}, **kwargs
)
print("fields stored = ", list(field_dict.keys()))
print(f"Gas Holdup = {gh:.4g}")
sup_vel, field_dict = compute_superficial_gas_velocity(
    volume_time="1", field_dict=field_dict, **kwargs
)
print("fields stored = ", list(field_dict.keys()))
print(f"Superficial velocity = {sup_vel:.4g} m/s")
y_ave_co2, field_dict = compute_ave_y_liq(
    volume_time="1", spec_name="CO2", field_dict=field_dict, **kwargs
)
print("fields stored = ", list(field_dict.keys()))
print(f"Reactor averaged YCO2 = {y_ave_co2:.4g}")
c_ave_co2, field_dict = compute_ave_conc_liq(
    volume_time="1",
    spec_name="CO2",
    mol_weight=0.04401,
    rho_val=1000,
    field_dict=field_dict,
    **kwargs,
)
print("fields stored = ", list(field_dict.keys()))
print(f"Reactor averaged [CO2] = {c_ave_co2:.4g} mol/m3")
diam, field_dict = compute_ave_bubble_diam(
    volume_time="1", field_dict=field_dict, **kwargs
)
print("fields stored = ", list(field_dict.keys()))
print(f"Reactor averaged bubble diameter = {diam:.4g} m")
