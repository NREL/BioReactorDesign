import sys

import numpy as np

sys.path.append("util")
from stlUtil import makeSpider, saveSTL

if __name__ == "__main__":
    # Spider
    combined = makeSpider(centerRad=0.25, nArms=12, widthArms=0.1, lengthArms=0.5)
    saveSTL(combined, filename="spg.stl")

    # plot
    import matplotlib.pyplot as plt
    from plotsUtil import *
    axes = plotSTL(combined)
    axprettyLabels(axes, "x", "", "z", 14)
    plt.show()
