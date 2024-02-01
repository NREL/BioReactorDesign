import os

from prettyPlot.plotting import *


def plotAll(data_files, data_arr, color_files=None, chop=False, extrap=False):
    fig = plt.figure()
    for idat, datf in enumerate(data_files):
        if color_files is not None:
            color = color_files[idat]
        else:
            color = None
        if extrap:
            plt.plot(
                data_arr[datf]["textrap"],
                100 * data_arr[datf]["yextrap"],
                "--",
                color=color,
            )
        if chop:
            lim = data_arr[datf]["lim"]
            plt.plot(
                data_arr[datf]["t"][lim:],
                100 * data_arr[datf]["y"][lim:],
                "o",
                color=color,
            )
        else:
            plt.plot(
                data_arr[datf]["t"], 100 * data_arr[datf]["y"], color=color
            )
    pretty_labels("time [s]", "yield [%]", 18)
    plt.show()


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth


dataRoot = "data"
color_files = ["k", "r", "b"]
data_files = ["240rpm_1e8", "150rpm_1e8", "150rpm_5e7"]
data_arr = {}

maxExp = 20


def f(x, a, b, c, d, e, f):
    return 2 * c * (
        0.5 - 1 / (1 + np.exp(np.clip((a * x) ** b, a_min=None, a_max=20)))
    ) + 2 * f * (
        0.5 - 1 / (1 + np.exp(np.clip((d * x) ** e, a_min=None, a_max=20)))
    )


from scipy.optimize import curve_fit

for idat, datf in enumerate(data_files):
    filename = os.path.join(dataRoot, datf)
    A = np.loadtxt(filename)
    data_arr[datf] = {}
    data_arr[datf]["t"] = A[:, 0]
    data_arr[datf]["y"] = A[:, 5] / (A[:, 4] * 16 / 44 + A[:, 5])
    increase_ind_arr = np.argwhere(np.diff(data_arr[datf]["y"]) > 0)
    increase_ind = increase_ind_arr[
        np.argwhere(data_arr[datf]["t"][increase_ind_arr] > 10)[0][0]
    ][0]
    print(data_arr[datf]["t"][increase_ind])
    data_arr[datf]["lim"] = increase_ind
    y_fit = (
        data_arr[datf]["y"][increase_ind:] - data_arr[datf]["y"][increase_ind]
    )
    t_fit = (
        data_arr[datf]["t"][increase_ind:] - data_arr[datf]["t"][increase_ind]
    )
    popt, pcov = curve_fit(
        f,
        t_fit,
        y_fit,
        bounds=([0, 0, 0, 0, 0, 0], [0.05, 1.7, np.inf, 0.015, 1.3, np.inf]),
    )
    data_arr[datf]["textrap"] = np.linspace(0, 600, 600)
    data_arr[datf]["yextrap"] = f(data_arr[datf]["textrap"], *popt)
    print(popt)
    data_arr[datf]["textrap"] += data_arr[datf]["t"][increase_ind]
    data_arr[datf]["yextrap"] += data_arr[datf]["y"][increase_ind]

plotAll(data_files, data_arr, color_files=color_files, chop=True, extrap=True)
# plotAll(data_files, data_arr,  chop=True, extrap=True)
