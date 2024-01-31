import sys
import numpy as np
from brd.meshing.stl_mesh_tools import makeSpider, saveSTL
from brd.utilities.stl_plotting import plotSTL, plt, pretty_labels

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
    axes = plotSTL("spg.stl")
    pretty_labels("x", "y", zlabel="z", fontsize=14, ax=axes)
