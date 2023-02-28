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
 


    # Rotated rectangle
    #rect_meshInpt1 = makeRectangle(0.25/10,0.25/10, shiftw=1, shifth=1)
    #rect1 = traceMesh(rect_meshInpt1)
    #rect1 = rotate(rect1, np.pi/3)
    #axes=plotSTL(rect1)
    #axprettyLabels(axes,'x','','z',14)
    #plt.show()





 
    # POLYGON 
    #polygon_meshInpt = makePolygon(rad=0.25,nvert=8) 
    #polygon = traceMesh(polygon_meshInpt)
    #axes=plotSTL(polygon)
    #axprettyLabels(axes,'x','','z',14)
    #plt.show()







 
    # CROSS
    #cross_meshInpt = makeCross()
    #cross = traceMesh(cross_meshInpt)
    #axes=plotSTL(cross)
    #axprettyLabels(axes,'x','','z',14)
    #plt.show()



    # CROSS
    #rect_meshInpt1 = makeRectangle(0.25/10,0.25/10)
    #rect_meshInpt2 = makeRectangle(0.25/10,0.25/10, shiftw=(0.25/10), shifth=0)
    #rect_meshInpt3 = makeRectangle(0.25/10,0.25/10, shiftw=-(0.25/10), shifth=0)
    #rect_meshInpt4 = makeRectangle(0.25/10,0.25/10, shiftw=0, shifth=(0.25/10))
    #rect_meshInpt5 = makeRectangle(0.25/10,0.25/10, shiftw=0, shifth=-(0.25/10))
    #rect1 = traceMesh(rect_meshInpt1)
    #rect2 = traceMesh(rect_meshInpt2)
    #rect3 = traceMesh(rect_meshInpt3)
    #rect4 = traceMesh(rect_meshInpt4)
    #rect5 = traceMesh(rect_meshInpt5)

    #combined = mesh.Mesh(np.concatenate([rect1.data, rect2.data, rect3.data, rect4.data, rect5.data]))

    #axes=plotSTL(combined)
    #axprettyLabels(axes,'x','','z',14)
    #plt.show()
   
    #saveSTL(combined) 





