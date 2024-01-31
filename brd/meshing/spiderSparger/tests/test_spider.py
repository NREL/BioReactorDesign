import sys

import numpy as np

sys.path.append("../util")
import argument
import matplotlib.pyplot as plt
from plotsUtil import *
from stlUtil import makeSpider, saveSTL


def test_spider():
    # Spider
    combined, globalArea = makeSpider(
        centerRad=0.25,
        nArms=12,
        widthArms=0.1,
        lengthArms=0.5,
    )
    print(f"\tglobalArea = {globalArea}")
    saveSTL(combined, filename="spg.stl")

    # plot
    axes = plotSTL(combined)
    axprettyLabels(axes, "x", "", "z", 14)
