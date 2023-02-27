import numpy as np
import stl
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import sys
sys.path.append('util')
from plotsUtil import *
from stlUtil import *


cross_meshInpt = makeCross()
cross = traceMesh(cross_meshInpt)

axes=plotSTL(cross)
axprettyLabels(axes,'x','','z',14)
plt.show()
