""" Definitions for Gaussian Processes models """
import gpytorch
import numpy as np

class ExactGPModel(gpytorch.models.ExactGP):
    """Exact GP model definition"""
    def __init__(self, train_x, train_y, ard_dim, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()

        # lengthscale_prior = gpytorch.priors.UniformPrior(a, b)

        self.covar_module = gpytorch.kernels.ScaleKernel(
                            # gpytorch.kernels.MaternKernel(nu=0.5,ard_num_dims=ard_dim))
                            gpytorch.kernels.RBFKernel(ard_num_dims=ard_dim, active_dims=np.arange(ard_dim)))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)