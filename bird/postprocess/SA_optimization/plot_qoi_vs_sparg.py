import numpy as np
from prettyPlot.plotting import *

nsparg = [1, 2, 3, 4, 5, 6, 7, 8]
qoi_opt = [13.485, 14.276, 14.635, 15.692, 16.072, 17.657, 19.349, 20.712]

fig = plt.figure()
plt.plot(nsparg, qoi_opt, linewidth=3, color="k")
pretty_labels(
    r"N$_{\rm sparg}$", r"Optimal QOI [kg$^2$/kWh$^2$]", 20, fontname="Times"
)
plt.savefig("marginal_gain.png")
plt.savefig("marginal_gain.pdf")
