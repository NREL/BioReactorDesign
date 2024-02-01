import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


def rbf_kernel(x1, x2, length_scale, amplitude):
    assert len(x1.shape) == 2
    assert len(x2.shape) == 2
    x1 = jnp.reshape(x1, (-1, 1, jnp.shape(x1)[1]))
    x2 = jnp.reshape(x2, (1, -1, jnp.shape(x2)[1]))
    if isinstance(length_scale, float):
        length_scale = (
            jnp.ones((jnp.shape(x1)[0], jnp.shape(x2)[0], jnp.shape(x1)[1]))
            * length_scale
        )
    elif len(length_scale) == 1:
        length_scale = (
            jnp.ones((jnp.shape(x1)[0], jnp.shape(x2)[0], jnp.shape(x1)[1]))
            * length_scale[0]
        )
    else:
        length_scale = jnp.ones(
            (jnp.shape(x1)[0], jnp.shape(x2)[0], jnp.shape(x1)[1])
        ) * jnp.reshape(jnp.array(length_scale), (1, 1, -1))
    pairwise_sq_diff = jnp.sum((x2 - x1) ** 2 / length_scale**2, axis=2)
    return amplitude * jnp.exp(-0.5 * pairwise_sq_diff)


def make_RBFutility_jax(
    x_data, y, length_scale=1.0, amplitude=1.0, obs_noise=0.01
):
    # Calculate the RBF matrix
    rbf_matrix_data = rbf_kernel(x_data, x_data, length_scale, amplitude)
    rbf_matrix_data = rbf_matrix_data + obs_noise * jnp.eye(
        jnp.shape(x_data)[0]
    )

    # Solve the linear system
    weights = jnp.linalg.solve(rbf_matrix, y)

    return weights


def rbf_interpolation_jax(
    x_data, y, x_interp, length_scale=1.0, amplitude=1.0, obs_noise=0.01
):
    rbf_matrix_data = rbf_kernel(x_data, x_data, length_scale, amplitude)
    rbf_matrix_data = rbf_matrix_data + obs_noise * jnp.eye(
        jnp.shape(x_data)[0]
    )

    # Solve the linear system
    weights = jnp.linalg.solve(rbf_matrix_data, y)

    # Mean
    rbf_matrix_interp = rbf_kernel(x_interp, x_data, length_scale, amplitude)
    mean_interp = rbf_matrix_interp @ weights

    # Uncertainty
    rbf_matrix_interp_interp = rbf_kernel(
        x_interp, x_interp, length_scale, amplitude
    )
    cov_interp = (
        rbf_matrix_interp_interp
        - rbf_matrix_interp
        @ jnp.linalg.solve(rbf_matrix_data, rbf_matrix_interp.T)
    )
    std_interp = jnp.sqrt(jnp.diag(cov_interp))

    return mean_interp, std_interp


if __name__ == "__main__":
    ndim = 2

    if ndim == 1:
        X = np.reshape(np.linspace(0, 10, 8), (-1, 1))
        X_interp = np.reshape(np.linspace(0, 10, 100), (-1, 1))
        y = np.reshape(np.squeeze(X * np.sin(2 * X)), (-1, 1))

        kernel = 1 * RBF(
            length_scale=[1.0] * ndim, length_scale_bounds=[(1, 1e2)] * ndim
        )
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50)
        gpr.fit(X, y)
        mean_prediction, std_prediction = gpr.predict(
            X_interp, return_std=True
        )
        std_prediction = np.reshape(std_prediction, mean_prediction.shape)

        gp_params = gpr.kernel_.get_params()
        length = gp_params["k2__length_scale"]
        amp = gp_params["k1__constant_value"]
        mean_pred_jax, std_pred_jax = rbf_interpolation_jax(
            X,
            y,
            X_interp,
            length_scale=length,
            amplitude=amp,
            obs_noise=0,
        )
        std_pred_jax = np.reshape(std_pred_jax, mean_pred_jax.shape)

        fig = plt.figure()
        plt.plot(X_interp, mean_prediction, color="b", label="sklearn")
        plt.plot(
            X_interp,
            mean_prediction - std_prediction,
            "--",
            color="b",
            linewidth=1,
        )
        plt.plot(
            X_interp,
            mean_prediction + std_prediction,
            "--",
            color="b",
            linewidth=1,
        )
        plt.plot(
            X_interp, mean_pred_jax, "s", markersize=10, color="r", label="jax"
        )
        plt.plot(
            X_interp,
            mean_pred_jax - std_pred_jax,
            "s",
            markersize=5,
            color="r",
        )
        plt.plot(
            X_interp,
            mean_pred_jax + std_pred_jax,
            "s",
            markersize=5,
            color="r",
        )
        plt.plot(X, y, "o")
        plt.legend()
        plt.show()

    if ndim == 2:
        # Input data points
        x = np.array([1, 2, 3, 4])
        y = np.array([1, 3, 2, 4])
        z = np.array([6, 7, 6, 8])
        xy = np.hstack((np.reshape(x, (-1, 1)), np.reshape(y, (-1, 1))))

        # Interpolation points
        x_interp = np.linspace(1, 4, 100)
        y_interp = np.linspace(1, 4, 100)
        xx_interp, yy_interp = np.meshgrid(x_interp, y_interp)
        xy_interp = np.hstack(
            (np.reshape(xx_interp, (-1, 1)), np.reshape(yy_interp, (-1, 1)))
        )

        kernel = 1 * RBF(
            length_scale=[2.0] * ndim, length_scale_bounds=[(1e-5, 1e5)] * ndim
        )
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)
        gpr.fit(xy, z)
        mean_prediction, std_prediction = gpr.predict(
            xy_interp, return_std=True
        )

        gp_params = gpr.kernel_.get_params()
        length = gp_params["k2__length_scale"]
        amp = gp_params["k1__constant_value"]
        mean_pred_jax, _ = rbf_interpolation_jax(
            xy,
            z,
            xy_interp,
            length_scale=length,
            amplitude=amp,
            obs_noise=0,
        )

        # Reshape the interpolated z values
        mean_prediction = mean_prediction.reshape(xx_interp.shape)
        mean_pred_jax = mean_pred_jax.reshape(xx_interp.shape)

        # Plot the original data points and interpolated surface
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(
            xx_interp, yy_interp, mean_prediction, cmap="Blues", alpha=0.8
        )
        ax.plot_surface(
            xx_interp, yy_interp, mean_pred_jax, cmap="Reds", alpha=0.8
        )
        ax.scatter(x, y, z, color="r", label="Data Points")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.legend()
        plt.show()
