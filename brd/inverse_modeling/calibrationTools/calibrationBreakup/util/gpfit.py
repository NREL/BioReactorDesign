import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel


def gpfit_simple(x, y, ndim):
    kernel = ConstantKernel(
        constant_value=[1], constant_value_bounds=[(1e-4, 1e3)]
    ) * RBF(
        length_scale=[1e1] * ndim, length_scale_bounds=[(1e-3, 1e3)] * ndim
    )
    gpr = GaussianProcessRegressor(
        kernel=kernel, alpha=0.0, n_restarts_optimizer=100
    )
    gpr.fit(x, y)
    y_mean, y_std = gpr.predict(x, return_std=True)
    like = gpr.log_marginal_likelihood(gpr.kernel_.theta)
    return y_mean, y_std, gpr, like
