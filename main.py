import sys

import numpy as np

sys.path.append("util")
import argument
from stlUtil import makeSpider, saveSTL

if __name__ == "__main__":
    args = argument.initArgs()

    # Spider
    combined, globalArea = makeSpider(
        centerRad=args.centerRadius,
        nArms=args.nArms,
        widthArms=args.armsWidth,
        lengthArms=args.armsLength,
    )
    print(f"\tglobalArea = {globalArea}")

    saveSTL(combined, filename="spg.stl")

    if args.verbose:
        # plot
        import matplotlib.pyplot as plt
        from plotsUtil import *

        axes = plotSTL(combined)
        axprettyLabels(axes, "x", "", "z", 14)
        plt.show()
