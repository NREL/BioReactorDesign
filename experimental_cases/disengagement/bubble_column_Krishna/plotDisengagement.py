import matplotlib.pyplot as plt
import pandas as pd

# Path to the file
file_path = "H.dat"

# Read the data from the file
df = pd.read_csv(file_path, delim_whitespace=True, comment='#', names=["time", "phase_com", "disengaged"])

# Plot phase_com vs. time
plt.figure(figsize=(8, 5))
plt.plot(df["time"], df["phase_com"], marker='o', linestyle='-', label="phase_com")
plt.xlabel("t [s]")
plt.ylabel("H")
#plt.title("Phase Center of Mass Over Time")
#plt.grid(True)
#plt.legend()
plt.show()

