import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

X = np.reshape(np.linspace(0, 10, 10), (-1,1))
X_interp = np.reshape(np.linspace(0, 10, 100), (-1,1))
y = np.reshape(np.squeeze(X * np.sin(X)), (-1,1))

kernel = 1*RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)
gpr.fit(X, y)
mean_prediction, std_prediction = gpr.predict(X_interp, return_std=True)
print(gpr.kernel_)

def rbf_interpolation_jax(x_data, y, x_interp, length_scale=1, amplitude=1):
    """Perform 2D point interpolation using squared exponential kernel."""
    def rbf_kernel(x1, x2):
        x1 = jnp.reshape(x1,(-1,1,jnp.shape(x1)[1]))
        x2 = jnp.reshape(x2,(1,-1,jnp.shape(x2)[1]))
        pairwise_sq_diff = jnp.sum((x2-x1)**2, axis=2)
        return amplitude * jnp.exp(-0.5 * pairwise_sq_diff / length_scale**2)

    # Calculate the RBF matrix
    rbf_matrix = rbf_kernel(x_data, x_data)
    # Solve the linear system
    weights = jnp.linalg.solve(rbf_matrix, y)

    # Interpolate the y values
    z_interp = jnp.einsum('ij,j->i', rbf_kernel(x_interp, x_data), weights)
    return z_interp


mean_pred_jax = rbf_interpolation_jax(X, np.squeeze(y), X_interp, length_scale=2.34, amplitude=9.78**2)

fig = plt.figure()
plt.plot(X_interp, mean_prediction, label='sklearn')
plt.plot(X_interp, mean_pred_jax, 's', label='jax')
plt.plot(X, y, 'o')
plt.legend()
plt.show()
