import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits import mplot3d


def plotSTL(mesh):
    fig = plt.figure()
    axes = mplot3d.Axes3D(fig)
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh.vectors))

    min_x = np.amin(mesh.points[:,0])
    max_x = np.amax(mesh.points[:,0])
    min_y = np.amin(mesh.points[:,1])
    max_y = np.amax(mesh.points[:,1])
    min_z = np.amin(mesh.points[:,2])
    max_z = np.amax(mesh.points[:,2])

    amp = np.array([max_x-min_x, max_y - min_y, max_z - min_z])
    eps = np.amax(amp)

    axes.set_xlim3d(left=min_x-eps, right=max_x+eps)
    axes.set_ylim3d(min_y-eps, max_y+eps)
    axes.set_zlim3d(min_z-eps, max_z+eps)

    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    plt.locator_params(axis='z', nbins=4)
  
    #2D view 
    if abs(amp[0])<1e-12:
        axes.view_init(0,90)
    elif abs(amp[1])<1e-12:
        axes.view_init(0,90)
        plt.yticks([])
    elif abs(amp[2])<1e-12:
        axes.view_init(0,90)
    

      
    return axes

def axprettyLabels(ax, xlabel, ylabel, zlabel, fontsize, title=None):
    ax.set_xlabel(
        xlabel,
        fontsize=fontsize,
        fontweight="bold",
        fontname="Times New Roman",
    )
    ax.set_ylabel(
        ylabel,
        fontsize=fontsize,
        fontweight="bold",
        fontname="Times New Roman",
    )
    ax.set_zlabel(
        zlabel,
        fontsize=fontsize,
        fontweight="bold",
        fontname="Times New Roman",
    )
    if not title == None:
        ax.set_title(
            title,
            fontsize=fontsize,
            fontweight="bold",
            fontname="Times New Roman",
        )
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontname("Times New Roman")
        tick.label1.set_fontweight("bold")
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontname("Times New Roman")
        tick.label1.set_fontweight("bold")
    for tick in ax.zaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontname("Times New Roman")
        tick.label1.set_fontweight("bold")
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color("black")
    plt.grid(color="k", linestyle="-", linewidth=0.5)
    try:
        plt.tight_layout()
    except:
        print("Could not call tight_layout")
        pass


def plotLegend():
    fontsize = 16
    plt.legend()
    leg = plt.legend(
        prop={
            "family": "Times New Roman",
            "size": fontsize - 3,
            "weight": "bold",
        }
    )
    leg.get_frame().set_linewidth(2.0)
    leg.get_frame().set_edgecolor("k")


