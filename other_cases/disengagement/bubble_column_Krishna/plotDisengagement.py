import matplotlib.pyplot as plt
import pandas as pd

# Path to the file
file_path = "./postProcessing/disengagement/0/disengagement.dat"

# Read the data from the file
df = pd.read_csv(file_path, delim_whitespace=True, comment='#', names=["time", "phase_com", "pressure", "disengaged"])

# Plot phase_com vs. time
plt.figure(figsize=(8, 5))
plt.plot(df["time"], df["phase_com"], marker='o', linestyle='-', label="phase_com")
plt.xlabel("t [s]")
plt.ylabel("H [m]")
#plt.title("Phase Center of Mass Over Time")
#plt.grid(True)
#plt.legend()
plt.show()

# Plot pressure vs. time
plt.figure(figsize=(8, 5))
plt.plot(df["time"], df["pressure"], marker='x', linestyle='-', label="pressure")
plt.xlabel("t [s]")
plt.ylabel("p [Pa]")
#plt.title("Phase Center of Mass Over Time")
#plt.grid(True)
#plt.legend()
plt.show()

