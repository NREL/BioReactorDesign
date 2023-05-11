import sys

import numpy as np
from scipy.optimize import minimize


def bounded_constraint(x):
    xend = 1 - np.sum(x)
    cons = 0
    for i in range(len(x)):
        cons += np.clip(-x[i], a_min=0, a_max=None)
        cons += np.clip((x[i] - 1), a_min=0, a_max=None)
    cons += np.clip(-xend, a_min=0, a_max=None)
    cons += np.clip((xend - 1), a_min=0, a_max=None)
    return 100 * cons


def std_constraint(x, meanTar, stdTar, diam):
    # x = np.clip(x, a_min=0, a_max=1)
    xend = 1 - np.sum(x)
    var = np.sum(x * (diam[:-1] - meanTar) ** 2)
    var += xend * (diam[-1] - meanTar) ** 2
    return 1e6 * abs(np.sqrt(np.clip(var, a_min=0, a_max=None)) - stdTar)


def mean_constraint(x, meanTar, diam):
    # x = np.clip(x, a_min=0, a_max=1)
    xend = 1 - np.sum(x)
    mean = np.sum(x * diam[:-1])
    mean += xend * diam[-1]
    return 1e6 * abs(mean - meanTar)


def bell_constraint(x, meanTar, diam):
    # x = np.clip(x, a_min=0, a_max=1)
    xend = 1 - np.sum(x)
    peakInd = np.argsort(abs(diam - meanTar))[0]
    cons = 0
    for i in range(peakInd):
        cons += np.clip(x[i] - x[i + 1], a_min=0, a_max=None)
    for i in range(peakInd + 1, len(x)):
        cons += np.clip(x[i] - x[i - 1], a_min=0, a_max=None)
    cons += np.clip(xend - x[-1], a_min=0, a_max=None)
    return 100 * cons


def objective(x, meanTar, stdTar, diam):
    return (
        bounded_constraint(x)
        + std_constraint(x, meanTar, stdTar, diam)
        + mean_constraint(x, meanTar, diam)
        + bell_constraint(x, meanTar, diam)
    )


def opt(meanTar, stdTar, diam):
    # res = minimize(objective, x0=0.33*np.ones(len(diam)-1), method="SLSQP")
    res = minimize(
        objective,
        args=(meanTar, stdTar, diam),
        x0=np.random.uniform(low=0, high=1, size=len(diam) - 1),
        method="SLSQP",
    )
    return res


def get_f_vals(meanTar, stdTar, diam, verb=True):
    if meanTar < np.amin(diam) or meanTar > np.amax(diam):
        sys.exit(
            f"ERROR: mean target {meanTar} out of bounds [{np.amin(diam)}, {np.amax(diam)}]"
        )
    tol = 10
    irep = 0
    while tol > 0.01:
        if verb:
            print(f"Rep = {irep}, tol={tol}")
        res = opt(meanTar, stdTar, diam)
        x = np.zeros(len(res.x) + 1)
        x[:-1] = res.x
        x[-1] = 1 - np.sum(res.x)
        x = np.clip(x, a_min=0, a_max=None)
        meanOut = np.sum(x * diam)
        stdOut = np.sqrt(
            np.sum(np.clip(x, a_min=0, a_max=None) * (diam - meanTar) ** 2)
        )
        tol = (
            abs(meanOut - meanTar) / meanTar
            + 0.1 * abs(stdOut - stdTar) / stdTar
        )
        irep += 1
        if irep > 100:
            sys.exit(
                "ERROR: optimization fail, typically occurs because the population balance domain is too tight"
            )

    if verb:
        print("meanTar = ", meanTar)
        print("stdTar = ", stdTar)
        print("x = ", x)
        print("mean = ", np.sum(x * diam))
        print(
            "std = ",
            np.sqrt(
                np.sum(np.clip(x, a_min=0, a_max=None) * (diam - meanTar) ** 2)
            ),
        )
    return x


def poreDiamCorr(dp, ds, Ugs):
    g = 9.81
    sigmaL = 0.07
    rhoL = 1000
    muL = 8.90e-4
    Fr = Ugs**2 / (ds * g)
    We = rhoL * Ugs**2 * ds / sigmaL
    Re = rhoL * Ugs * ds / muL
    d32 = (
        ds
        * 7.35
        * (We ** (-1.7) * Re ** (0.1) * Fr ** (1.8) * (dp / ds) ** (1.7))
        ** 0.2
    )
    return d32, d32 * 0.15


if __name__ == "__main__":
    meanTar, stdTar = poreDiamCorr(2e-5, 0.15, 0.01)
    print(meanTar)
    print(stdTar)
    # meanTar = 2.1e-3
    # stdTar = meanTar*0.15
    diam = np.linspace(0.5e-3, 6e-3, 10)
    x = get_f_vals(meanTar, stdTar, diam)

    import matplotlib.pyplot as plt

    plt.plot(diam, x, "o")
    plt.show()
