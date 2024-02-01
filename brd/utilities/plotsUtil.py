import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# from SALib.analyze import delta


def prettyLabels(xlabel, ylabel, fontsize, title=None, grid=True):
    plt.xlabel(
        xlabel,
        fontsize=fontsize,
        fontweight="bold",
        fontname="Times New Roman",
    )
    plt.ylabel(
        ylabel,
        fontsize=fontsize,
        fontweight="bold",
        fontname="Times New Roman",
    )
    if not title == None:
        plt.title(
            title,
            fontsize=fontsize,
            fontweight="bold",
            fontname="Times New Roman",
        )
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontname("Times New Roman")
        tick.label1.set_fontweight("bold")
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontname("Times New Roman")
        tick.label1.set_fontweight("bold")
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color("black")
    if grid:
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


def snapVizZslice(field, x, y, figureDir, figureName, title=None):
    fig, ax = plt.subplots(1)
    plt.imshow(
        np.transpose(field),
        cmap=cm.jet,
        interpolation="bicubic",
        vmin=np.amin(field),
        vmax=np.amax(field),
        extent=[np.amin(x), np.amax(x), np.amax(y), np.amin(y)],
    )
    prettyLabels("x [m]", "y [m]", 16, title)
    plt.colorbar()
    fig.savefig(figureDir + "/" + figureName)
    plt.close(fig)
    return 0


def movieVizZslice(field, x, y, itime, movieDir, minVal=None, maxVal=None):
    fig, ax = plt.subplots(1)
    fontsize = 16
    if minVal == None:
        minVal = np.amin(field)
    if maxVal == None:
        maxVal = np.amax(field)
    plt.imshow(
        np.transpose(field),
        cmap=cm.jet,
        interpolation="bicubic",
        vmin=minVal,
        vmax=maxVal,
        extent=[np.amin(x), np.amax(x), np.amax(y), np.amin(y)],
    )
    prettyLabels("x [m]", "y [m]", 16, "Snap Id = " + str(itime))
    plt.colorbar()
    fig.savefig(movieDir + "/im_" + str(itime) + ".png")
    plt.close(fig)
    return 0


def makeMovie(ntime, movieDir, movieName):
    fig = plt.figure()
    # initiate an empty  list of "plotted" images
    myimages = []
    # loops through available png:s
    for i in range(ntime):
        ## Read in picture
        fname = movieDir + "/im_" + str(i) + ".png"
        myimages.append(imageio.imread(fname))
    imageio.mimsave(movieName, myimages)
    return


def plotHist(field, xLabel, folder, filename):
    fig = plt.figure()
    plt.hist(field)
    fontsize = 18
    prettyLabels(xLabel, "bin count", fontsize)
    fig.savefig(folder + "/" + filename)


def plotContour(x, y, z, color):
    ax = plt.gca()
    X, Y = np.meshgrid(x, y)
    CS = ax.contour(
        X, Y, np.transpose(z), [0.001, 0.005, 0.01, 0.05], colors=color
    )
    h, _ = CS.legend_elements()
    return h[0]


def plotActiveSubspace(paramName, W, title=None):
    x = []
    for i, name in enumerate(paramName):
        x.append(i)
    fig = plt.figure()
    plt.bar(
        x,
        W,
        width=0.8,
        bottom=None,
        align="center",
        data=None,
        tick_label=paramName,
    )
    fontsize = 16
    if not title == None:
        plt.title(
            title,
            fontsize=fontsize,
            fontweight="bold",
            fontname="Times New Roman",
        )
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontname("Times New Roman")
        tick.label1.set_fontweight("bold")
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontname("Times New Roman")
        tick.label1.set_fontweight("bold")
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color("black")
        # ax.spines[axis].set_zorder(0)
    plt.grid(color="k", linestyle="-", linewidth=0.5)
    try:
        plt.tight_layout()
    except:
        print("Could not call tight_layout")
        pass


def axprettyLabels(ax, xlabel, ylabel, fontsize, title=None, grid=True):
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
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color("black")
    if grid:
        ax.grid(color="k", linestyle="-", linewidth=0.5)
    try:
        plt.tight_layout()
    except:
        print("Could not call tight_layout")
        pass


def axplotLegend(ax):
    fontsize = 16
    ax.legend()
    leg = ax.legend(
        prop={
            "family": "Times New Roman",
            "size": fontsize - 3,
            "weight": "bold",
        }
    )
    leg.get_frame().set_linewidth(2.0)
    leg.get_frame().set_edgecolor("k")


def plotTrainingLogs(trainingLoss, validationLoss):
    fig = plt.figure()
    plt.plot(trainingLoss, color="k", linewidth=3, label="train")
    plt.plot(validationLoss, "--", color="k", linewidth=3, label="test")
    prettyLabels("epoch", "loss", 14, title="model loss")
    plotLegend()


def plotScatter(
    dataX, dataY, freq, title=None, xfeat=None, yfeat=None, fontSize=14
):
    fig = plt.figure()
    if xfeat is None:
        xfeat = 0
    if yfeat is None:
        yfeat = 1

    plt.plot(dataX[0::freq], dataY[0::freq], "o", color="k", markersize=3)
    if title is None:
        # prettyLabels('feature '+str(xfeat),'feature '+str(yfeat),fontSize)
        prettyLabels("", "", fontSize)
    else:
        prettyLabels("", "", fontSize, title=title)


def plot_probabilityMapDouble2D(
    model, minX, maxX, minY, maxY, nx=100, ny=100, minval=None, maxval=None
):
    x = np.linspace(minX, maxX, nx)
    y = np.linspace(minY, maxY, ny)
    sample = np.float32(np.zeros((nx, ny, 2)))
    for i in range(nx):
        for j in range(ny):
            sample[i, j, 0] = x[i]
            sample[i, j, 1] = y[j]
    sample = np.reshape(sample, (nx * ny, 2))
    prob = np.exp(model.log_prob(sample))
    prob = np.reshape(prob, (nx, ny))

    if minval is None:
        minval = np.amin(prob)
    if maxval is None:
        maxval = np.amax(prob)

    fig = plt.figure()
    plt.imshow(
        np.transpose(prob),
        cmap=cm.jet,
        interpolation="bicubic",
        vmin=minval,
        vmax=maxval,
        extent=[np.amin(x), np.amax(x), np.amax(y), np.amin(y)],
        aspect="auto",
    )
    plt.gca().invert_yaxis()
    plt.colorbar()
    prettyLabels(
        "1st label", "2nd label", 20, title="Approximate Probability Map"
    )


def plot_fromLatentToData(model, nSamples, xfeat=None, yfeat=None):
    if xfeat is None:
        xfeat = 0
    if yfeat is None:
        yfeat = 1
    samples = model.distribution.sample(nSamples)
    print(samples.shape)
    x, _ = model.predict(samples)
    f, axes = plt.subplots(1, 2)
    axes[0].plot(
        samples[:, xfeat], samples[:, yfeat], "o", markersize=3, color="k"
    )
    axprettyLabels(
        axes[0],
        "feature " + str(xfeat),
        "feature " + str(yfeat),
        14,
        title="Prior",
    )
    axes[1].plot(x[:, xfeat], x[:, yfeat], "o", markersize=3, color="k")
    axprettyLabels(
        axes[1],
        "feature " + str(xfeat),
        "feature " + str(yfeat),
        14,
        title="Generated",
    )


def scatter_BSD(
    listDatax,
    listDatat,
    listData,
    listTitle,
    listXAxisName=None,
    YAxisName=None,
    vminList=None,
    vmaxList=None,
    globalTitle=None,
    barLabelList=None,
):
    lim = -1
    lim_vmax_t = -1
    lim_vmax_x = -1
    lim_plot = -1
    fig, axs = plt.subplots(1, len(listData), figsize=(len(listData) * 4, 4))
    if len(listData) == 1:
        i_dat = 0
        data = listData[i_dat]
        data_x = np.squeeze(listDatax[i_dat])
        data_t = np.squeeze(listDatat[i_dat])
        if vminList == None:
            vmin = np.nanmin(data)
        else:
            try:
                vmin = vminList[i_dat]
            except:
                vmin = vminList[0]
        if vmaxList == None:
            vmax = np.nanmax(data) + 1e-10
        else:
            try:
                vmax = vmaxList[i_dat] + 1e-10
            except:
                vmax = vmaxList[0] + 1e-10
        cm = plt.cm.get_cmap("viridis")
        sc = axs.scatter(
            data_x, data_t, c=data, vmin=vmin, vmax=vmax, s=20, cmap=cm
        )
        # axs.invert_yaxis()
        divider = make_axes_locatable(axs)
        cax = divider.append_axes("right", size="10%", pad=0.2)
        cbar = fig.colorbar(sc, cax=cax)
        if not barLabelList is None:
            cbar.set_label(barLabelList[i_dat])
        ax = cbar.ax
        text = ax.yaxis.label
        font = matplotlib.font_manager.FontProperties(
            family="times new roman", weight="bold", size=14
        )
        text.set_font_properties(font)
        if i_dat == 0:
            axprettyLabels(
                axs,
                listXAxisName[i_dat],
                YAxisName,
                12,
                listTitle[i_dat],
                grid=False,
            )
        else:
            axprettyLabels(
                axs, listXAxisName[i_dat], "", 12, listTitle[i_dat], grid=False
            )

        ax.set_xticks([])  # values
        ax.set_xticklabels([])  # labels
        if not i_dat == 0:
            ax.set_yticks([])  # values
            ax.set_yticklabels([])  # labels
        for l in cbar.ax.yaxis.get_ticklabels():
            l.set_weight("bold")
            l.set_family("serif")
            l.set_fontsize(12)
    else:
        for i_dat in range(len(listData)):
            data = listData[i_dat]
            data_x = np.squeeze(listDatax[i_dat])
            data_t = np.squeeze(listDatat[i_dat])
            if vminList == None:
                vmin = np.nanmin(data)
            else:
                try:
                    vmin = vminList[i_dat]
                except:
                    vmin = vminList[0]
            if vmaxList == None:
                vmax = np.nanmax(data) + 1e-10
            else:
                try:
                    vmax = vmaxList[i_dat] + 1e-10
                except:
                    vmax = vmaxList[0] + 1e-10
            cm = plt.cm.get_cmap("viridis")
            sc = axs[i_dat].scatter(
                data_x, data_t, c=data, vmin=vmin, vmax=vmax, s=20, cmap=cm
            )
            # axs[i_dat].invert_yaxis()
            divider = make_axes_locatable(axs[i_dat])
            cax = divider.append_axes("right", size="10%", pad=0.2)
            cbar = fig.colorbar(sc, cax=cax)
            if not barLabelList is None:
                cbar.set_label(barLabelList[i_dat])
            ax = cbar.ax
            text = ax.yaxis.label
            font = matplotlib.font_manager.FontProperties(
                family="times new roman", weight="bold", size=14
            )
            text.set_font_properties(font)
            if i_dat == 0:
                axprettyLabels(
                    axs[i_dat],
                    listXAxisName[i_dat],
                    YAxisName,
                    12,
                    listTitle[i_dat],
                    grid=False,
                )
            else:
                axprettyLabels(
                    axs[i_dat],
                    listXAxisName[i_dat],
                    "",
                    12,
                    listTitle[i_dat],
                    grid=False,
                )

            if not i_dat == 0:
                axs[i_dat].set_yticks([])
                axs[i_dat].set_yticklabels([])
            for l in cbar.ax.yaxis.get_ticklabels():
                l.set_weight("bold")
                l.set_family("serif")
                l.set_fontsize(12)

        if not globalTitle is None:
            plt.subplots_adjust(top=0.85)
            plt.suptitle(
                globalTitle,
                fontsize=14,
                fontweight="bold",
                fontname="Times New Roman",
            )
    # plt.tight_layout()


def sobol_ind(params, X, Y, bounds, title=None):
    nData = len(Y)
    nDim = X.shape[1]
    # Sobol
    problem = {
        "num_vars": nDim,
        "names": params,
        "bounds": bounds,
    }
    Si = delta.analyze(problem, X, Y, print_to_console=True, seed=42)
    x = []
    for i, name in enumerate(params):
        x.append(i)
    fig = plt.figure(figsize=(4, 4))
    plt.bar(
        x,
        Si["S1"],
        yerr=Si["S1_conf"],
        alpha=1,
        ecolor="black",
        capsize=10,
        linewidth=3,
        width=0.8,
        bottom=None,
        align="center",
        data=None,
        tick_label=params,
    )
    fontsize = 18
    plt.ylabel(
        "Sobol indices",
        fontsize=fontsize,
        fontweight="bold",
        fontname="Times New Roman",
        labelpad=0,
    )
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontname("Times New Roman")
        tick.label1.set_fontweight("bold")
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontname("Times New Roman")
        tick.label1.set_fontweight("bold")
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color("black")
        # ax.spines[axis].set_zorder(0)
    plt.grid(color="k", linestyle="-", linewidth=0.5)
    if not title is None:
        plt.title(
            title,
            fontsize=fontsize,
            fontweight="bold",
            fontname="Times New Roman",
        )
    plt.tight_layout()


def ax_sobol_ind(ax, params, X, Y, bounds, title=None):
    nData = len(Y)
    nDim = X.shape[1]
    # Sobol
    problem = {
        "num_vars": nDim,
        "names": params,
        "bounds": bounds,
    }
    Si = delta.analyze(problem, X, Y, print_to_console=True, seed=42)
    x = []
    for i, name in enumerate(params):
        x.append(i)
    ax.bar(
        x,
        Si["S1"],
        yerr=Si["S1_conf"],
        alpha=1,
        ecolor="black",
        capsize=10,
        linewidth=3,
        width=0.8,
        bottom=None,
        align="center",
        data=None,
        tick_label=params,
    )
    ax.bar(
        x,
        Si["S1"],
        width=0.8,
        bottom=None,
        align="center",
        data=None,
        tick_label=params,
    )
    fontsize = 18
    ax.set_ylabel(
        "Sobol indices",
        fontsize=fontsize,
        fontweight="bold",
        fontname="Times New Roman",
        labelpad=0,
    )
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontname("Times New Roman")
        tick.label1.set_fontweight("bold")
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontname("Times New Roman")
        tick.label1.set_fontweight("bold")
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color("black")
        # ax.spines[axis].set_zorder(0)
    ax.grid(color="k", linestyle="-", linewidth=0.5)
    if not title is None:
        ax.set_title(
            title,
            fontsize=fontsize,
            fontweight="bold",
            fontname="Times New Roman",
        )


def label_conv(input_string):
    if input_string.lower() == "width":
        return "width [mm]"
    elif input_string.lower() == "spacing":
        return "spacing [mm]"
    elif input_string.lower() == "height":
        return "height [mm]"
    elif (
        input_string.lower() == "co2_liq"
        or input_string.lower() == "co2.liquid"
    ):
        return r"$Y_{CO_2, liq}$"
    elif (
        input_string.lower() == "co_liq" or input_string.lower() == "co.liquid"
    ):
        return r"$Y_{CO, liq}$"
    elif (
        input_string.lower() == "h2_liq" or input_string.lower() == "h2.liquid"
    ):
        return r"$Y_{H_2, liq}$"
    elif (
        input_string.lower() == "co2_gas" or input_string.lower() == "co2.gas"
    ):
        return r"$Y_{CO_2, gas}$"
    elif input_string.lower() == "co_gas" or input_string.lower() == "co.gas":
        return r"$Y_{CO, gas}$"
    elif input_string.lower() == "h2_gas" or input_string.lower() == "h2.gas":
        return r"$Y_{H_2, gas}$"
    elif input_string.lower() == "kla_h2":
        return r"$KLA_{H_2}$"
    elif input_string.lower() == "kla_co":
        return r"$KLA_{CO}$"
    elif input_string.lower() == "kla_co2":
        return r"$KLA_{CO_2}$"
    elif input_string.lower() == "alpha.gas":
        return r"$\alpha_{gas}$"
    elif (
        input_string.lower() == "d.gas"
        or input_string.lower() == "d"
        or input_string.lower() == "bubblediam"
    ):
        return "Mean bubble diam [m]$"
    elif input_string.lower() == "y":
        return "y [m]"
    elif input_string.lower() == "t":
        return "t [s]"
    elif input_string.lower() == "gh":
        return "Gas holdup"
    elif input_string.lower() == "gh_height":
        return "Height-based gas holdup"
    else:
        print(input_string)
        return input_string


def plot_bar_names(params, val, title=None):
    x = []
    for i, name in enumerate(params):
        x.append(i)
    plt.bar(
        x,
        val,
        width=0.8,
        bottom=None,
        align="center",
        data=None,
        tick_label=params,
    )
    fontsize = 16
    if not title == None:
        plt.title(
            label_conv(title),
            fontsize=fontsize,
            fontweight="bold",
            fontname="Times New Roman",
        )
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontname("Times New Roman")
        tick.label1.set_fontweight("bold")
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontname("Times New Roman")
        tick.label1.set_fontweight("bold")
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color("black")
        # ax.spines[axis].set_zorder(0)
    plt.grid(color="k", linestyle="-", linewidth=0.5)
    plt.tight_layout()
