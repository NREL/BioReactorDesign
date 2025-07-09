import numpy as np
from prettyPlot.plotting import plt, pretty_labels, pretty_legend


def plotSTL(stl_file):
    from mpl_toolkits import mplot3d
    from stl import mesh

    # Create a new plot
    figure = plt.figure()
    axes = figure.add_subplot(projection="3d")

    # Load the STL files and add the vectors to the plot
    your_mesh = mesh.Mesh.from_file(stl_file)

    poly_collection = mplot3d.art3d.Poly3DCollection(your_mesh.vectors)
    poly_collection.set_color((0, 0, 0))
    axes.add_collection3d(poly_collection)

    min_x = np.amin(your_mesh.points[:, 0])
    max_x = np.amax(your_mesh.points[:, 0])
    min_y = np.amin(your_mesh.points[:, 1])
    max_y = np.amax(your_mesh.points[:, 1])
    min_z = np.amin(your_mesh.points[:, 2])
    max_z = np.amax(your_mesh.points[:, 2])

    amp = np.array([max_x - min_x, max_y - min_y, max_z - min_z])
    # 2D view
    if abs(amp[0]) < 1e-12:
        axes.view_init(0, 90)
    elif abs(amp[1]) < 1e-12:
        axes.view_init(0, 90)
        plt.yticks([])
    elif abs(amp[2]) < 1e-12:
        axes.view_init(0, 90)

    eps = np.amax(amp)
    axes.set_xlim3d(left=min_x - eps, right=max_x + eps)
    axes.set_ylim3d(min_y - eps, max_y + eps)
    axes.set_zlim3d(min_z - eps, max_z + eps)
    plt.locator_params(axis="x", nbins=4)
    plt.locator_params(axis="y", nbins=4)
    plt.locator_params(axis="z", nbins=4)

    return axes
