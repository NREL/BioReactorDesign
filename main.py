import numpy as np
import stl
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import sys
sys.path.append('util')
from plotsUtil import *
from stlUtil import *



if __name__=="__main__":
 

    # Spider
    combined = makeSpider(centerRad=0.25, nArms=12, widthArms=0.1, lengthArms=0.5)
    saveSTL(combined) 
    
    # plot
    axes=plotSTL(combined)
    axprettyLabels(axes,'x','','z',14)
    plt.show()





