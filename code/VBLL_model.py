# VBLL_model.py

"""
This module implements the core components for a Variational Bayesian Last Layer,
enabling uncertainty quantification in neural networks.

**This code is modified from the official implementation accompanying the paper:**
  J. Harrison, J. Willes, and J. Snoek, “Variational Bayesian last layers,”
  in Proc. 12th Int. Conf. Learn. Represent., Vienna, Austria, May 7-11, 2024.

Utility functions for Cholesky updates (`cholesky_inverse`, `cholupdate`) are
credited to the fannypack library.

This file provides:
- Custom Gaussian distribution classes with different covariance parameterizations.
- The `VBLL_Layer`, which can replace the final layer of a standard NN.
- Helper functions for calculating uncertainty-aware metrics like ECE.
"""

import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Utility Functions ---

# Credit to fannypack for cholesky_inverse and cholupdate:
# https://github.com/brentyi/fannypack/blob/master/fannypack/utils/_math.py

def cholesky_inverse(u: torch.Tensor, upper: bool = False) -> torch.Tensor:
    """
    Computes the inverse of a matrix given its Cholesky factor. Supports batch dimensions.
    """
    if u.dim() == 2 and not u.requires_grad:
        # Use PyTorch's native function for simple cases
        return torch.cholesky_inverse(u, upper=upper)
    
    # Solve for the identity matrix to get the inverse
    identity = torch.eye(u.size(-1), device=u.device, dtype=u.dtype).expand_as(u)
    return torch.cholesky_solve(identity, u, upper=upper)

def cholupdate(L: torch.Tensor, x: torch.Tensor, weight: Optional[Union[torch.Tensor, float]] = 1.0) -> torch.Tensor:
    """
    Performs a batched rank-1 Cholesky update: L'L'^T = LL^T + weight * xx^T.
    """
    # Flatten batch dimensions for iterative update
    batch_dims = L.shape[:-2]
    matrix_dim = x.shape[-1]
    L = L.reshape(-1, matrix_dim, matrix_dim)
    x = x.reshape(-1, matrix_dim).clone()
    
    # Handle weight sign and magnitude
    sign = torch.sign(torch.tensor(weight)) if isinstance(weight, float) else torch.sign(weight)
    x = x * torch.sqrt(torch.abs(torch.tensor(weight, device=x.device, dtype=x.dtype)))

    # Iterative Cholesky update algorithm
    for k in range(matrix_dim):
        r_sq = L[:, k, k] ** 2 + sign * x[:, k] ** 2
        # Clamp to avoid numerical instability with negative updates
        r = torch.sqrt(torch.clamp(r_sq, min=1e-8))
        c = (r / L[:, k, k]).unsqueeze(-1)
        s = (x[:, k] / L[:, k, k]).unsqueeze(-1)
        
        # Update the rest of the matrix
        if k < matrix_dim - 1:
            L[:, k+1:, k] = (L[:, k+1:, k] + sign * s * x[:, k+1:]) / c
            x[:, k+1:] = c * x[:, k+1:] - s * L[:, k+1:, k]
        L[:, k, k] = r

    return L.reshape(batch_dims + (matrix_dim, matrix_dim))

def tp(M):
    """Shorthand for transpose of the last two dimensions."""
    return M.transpose(-1, -2)

# --- Custom Distribution Classes ---

class Normal(torch.distributions.Normal):
    """A Normal distribution with diagonal covariance."""
    @property
    def mean(self): return self.loc
    @property
    def var(self): return self.scale.pow(2)
    @property
    def covariance(self): return torch.diag_embed(self.var)
    @property
    def logdet_covariance(self): return 2 * torch.log(self.scale).sum(-1)

class DenseNormal(torch.distributions.MultivariateNormal):
    """A Normal distribution with a full-rank covariance matrix, parameterized by Cholesky factor."""
    def __init__(self, loc, cholesky):
        super().__init__(loc, scale_tril=cholesky)
    @property
    def mean(self): return self.loc
    @property
    def chol_covariance(self): return self.scale_tril
    @property
    def covariance(self): return self.scale_tril @ tp(self.scale_tril)
    @property
    def logdet_covariance(self): return 2. * torch.diagonal(self.scale_tril, dim1=-2, dim2=-1).log().sum(-1)

# Other distribution classes like LowRankNormal, DenseNormalPrec can be kept as is.
# ...

cov_param_dict = {
    'dense': DenseNormal,
    'diagonal': Normal,
}

def get_parameterization(p):
  if p in cov_param_dict:
    return cov_param_dict[p]
  else:
    raise ValueError(f"Invalid covariance parameterization '{p}'. Choose from {list(cov_param_dict.keys())}.")

def gaussian_kl(p, q_scale):
    """Computes the KL divergence between a Gaussian p and a scaled isotropic Gaussian q."""
    feat_dim = p.mean.shape[-1]
    mse_term = p.mean.pow(2).sum(-1).sum(-1) / q_scale
    trace_term = p.covariance.diagonal(dim1=-2, dim2=-1).sum(-1).sum(-1) / q_scale
    logdet_term = (feat_dim * np.log(q_scale) - p.logdet_covariance).sum(-1)

    return 0.5 * (mse_term + trace_term + logdet_term)

# --- Data Structure for VBLL Output ---

@dataclass
class VBLLReturn:
    """Data class to structure the output of the VBLL_Layer."""
    predictive: torch.distributions.Distribution
    train_loss_fn: Callable[[torch.Tensor], torch.Tensor]
    ood_scores: Optional[torch.Tensor] = None

# --- Main VBLL Layer ---

class VBLL_Layer(nn.Module):
    """
    A Variational Bayesian Last Layer.
    
    Replaces a standard final classification layer to provide uncertainty estimates
    through a Bayesian treatment of the layer's weights.
    """
    def __init__(self,
                 in_features,
                 out_features,
                 regularization_weight,
                 parameterization='dense',
                 prior_scale=1.,
                 wishart_scale=0.1,
                 dof=1.,
                 **kwargs):
        super().__init__()
        
        self.wishart_scale = wishart_scale
        self.dof = (dof + out_features + 1.) / 2.
        self.regularization_weight = regularization_weight

        # Prior distribution for the weights (zero mean, scaled identity covariance)
        self.prior_scale = prior_scale * (2. / in_features) # Kaiming-like scaling

        # Learnable noise parameters
        self.noise_mean = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        self.noise_logdiag = nn.Parameter(torch.randn(out_features) - 1)

        # Learnable parameters for the weight distribution
        self.W_dist_type = get_parameterization(parameterization)
        self.W_mean = nn.Parameter(torch.randn(out_features, in_features) * np.sqrt(2./in_features))
        
        if self.W_dist_type == DenseNormal:
            self.W_offdiag = nn.Parameter(1e-3 * torch.randn(out_features, in_features, in_features) / in_features)
            self.W_logdiag = nn.Parameter(1e-3 * torch.randn(out_features, in_features) - np.log(in_features))
        else: # Diagonal
            self.W_logdiag = nn.Parameter(1e-3 * torch.randn(out_features, in_features) - np.log(in_features))

    def W(self):
        """Constructs the weight distribution from its learnable parameters."""
        cov_diag = torch.exp(self.W_logdiag)
        if self.W_dist_type == DenseNormal:
            tril = torch.tril(self.W_offdiag, diagonal=-1) + torch.diag_embed(cov_diag)
            return self.W_dist_type(self.W_mean, tril)
        else: # Diagonal
            return self.W_dist_type(self.W_mean, cov_diag)

    def noise(self):
        """Constructs the noise distribution."""
        return Normal(self.noise_mean, torch.exp(self.noise_logdiag))

    def logit_predictive(self, x):
        """Computes the predictive distribution over logits."""
        # Linear transformation with uncertainty propagation
        return (self.W() @ x.unsqueeze(-1)).squeeze(-1) + self.noise()

    def jensen_bound(self, x, y):
        """Computes the Jensen's inequality-based lower bound on the log-likelihood."""
        pred = self.logit_predictive(x)
        linear_term = pred.mean.gather(1, y.unsqueeze(1)).squeeze(1)
        # Variance term in the bound
        var_term = pred.covariance.diagonal(dim1=-2, dim2=-1)
        pre_lse_term = pred.mean + 0.5 * var_term
        lse_term = torch.logsumexp(pre_lse_term, dim=-1)
        return linear_term - lse_term

    def _get_train_loss_fn(self, x):
        """Returns a callable loss function for training."""
        def loss_fn(y):
            # Evidence Lower Bound (ELBO)
            # 1. Expected Log-Likelihood (approximated by the bound)
            log_likelihood_bound = self.jensen_bound(x, y)
            
            # 2. KL Divergence between posterior and prior on weights
            kl_term = gaussian_kl(self.W(), self.prior_scale)
            
            # 3. Wishart prior term on noise precision
            noise = self.noise()
            noise_precision = torch.diag_embed(torch.exp(-2 * self.noise_logdiag))
            wishart_term = (self.dof * torch.logdet(noise_precision) - 0.5 * self.wishart_scale * torch.trace(noise_precision))
            
            # Combine terms, scaled by regularization weight
            elbo = log_likelihood_bound.mean() + self.regularization_weight * (wishart_term - kl_term.mean())
            
            # Return negative ELBO as the loss to be minimized
            return -elbo
        return loss_fn

    def forward(self, x):
        """
        Performs the forward pass and returns a structured output.
        """
        # The predictive distribution over class probabilities
        logit_pred = self.logit_predictive(x)
        # Using Categorical for standard classification output.
        # The `.probs` attribute will be the softmax of the mean logits.
        predictive_dist = torch.distributions.Categorical(logits=logit_pred.mean)
        
        # OOD score based on the max predictive probability
        ood_scores = torch.max(predictive_dist.probs, dim=-1)[0]
        
        return VBLLReturn(
            predictive=predictive_dist,
            train_loss_fn=self._get_train_loss_fn(x),
            ood_scores=ood_scores
        )

# --- ECE Calculation Utility ---
def ece_with_uncertainty(variance, probs, targets, n_bins=15, norm='l1'):
    """
    Calculates Expected Calibration Error (ECE), using inverse variance as confidence.
    
    Args:
        variance (torch.Tensor): Predictive variance tensor of shape (N,).
        probs (torch.Tensor): Predictive probabilities of shape (N,) for the positive class.
        targets (torch.Tensor): Ground truth labels of shape (N,), containing 0s and 1s.
        n_bins (int): Number of bins to use for ECE calculation.
        norm (str): Norm to use for the error ('l1' or 'l2').

    Returns:
        float: The calculated ECE value.
    """
    if not isinstance(targets, torch.Tensor): targets = torch.tensor(targets)
    if not isinstance(probs, torch.Tensor): probs = torch.tensor(probs)
    if not isinstance(variance, torch.Tensor): variance = torch.tensor(variance)
    
    predictions = (probs >= 0.5).long()
    
    # Use normalized inverse variance as the confidence score
    confidences = 1.0 / (variance + 1e-8)
    confidences = (confidences - confidences.min()) / (confidences.max() - confidences.min() + 1e-8)
    
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = (predictions[in_bin] == targets[in_bin]).float().mean()
            confidence_in_bin = confidences[in_bin].mean()
            
            error = torch.abs(accuracy_in_bin - confidence_in_bin)
            if norm == 'l2':
                error = error.pow(2)
            
            ece += prop_in_bin * error
            
    return ece.item()