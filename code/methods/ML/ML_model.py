# methods/ML/ML_model.py

"""
Defines the core Multi-Task Deep Neural Network (MT-DNN) architecture
for the Margin Likelihood (ML) training approach.

This module contains:
- A `Variation_Layer`, a simplified Bayesian layer that parameterizes weights
  and biases as Gaussian distributions.
- The `MT_DNN` class, which uses the `Variation_Layer` for its task heads
  and provides methods for calculating the ELBO and margin likelihood required
  for alternating optimization.
"""

import torch
from torch import nn
import torch.nn.functional as F

class Variation_Layer(nn.Module):
    """
    A simplified variational Bayesian layer for binary classification.
    
    It models weights and biases as independent Gaussians and uses a variational
    approach to approximate the posterior, enabling uncertainty estimation.
    """
    def __init__(self, in_fdim, out_fdim):
        super(Variation_Layer, self).__init__()
        self.in_fdim = in_fdim
        self.out_fdim = out_fdim

        # Parameters for the approximate posterior q(w) ~ N(w_mean, exp(w_log_var))
        self.w_mean = nn.Parameter(torch.randn(self.in_fdim, self.out_fdim))
        self.w_log_var = nn.Parameter(torch.randn(self.in_fdim, self.out_fdim))
        
        # Parameters for the approximate posterior q(b) ~ N(b_mean, exp(b_log_var))
        self.b_mean = nn.Parameter(torch.randn(self.out_fdim))
        self.b_log_var = nn.Parameter(torch.randn(self.out_fdim))
    
    def forward(self, x: torch.Tensor, return_var: bool = False):
        """Performs a forward pass, optionally returning predictive variance."""
        # Sample weights and biases from their distributions
        w = self.w_mean + torch.randn_like(self.w_mean) * torch.exp(0.5 * self.w_log_var)
        b = self.b_mean + torch.randn_like(self.b_mean) * torch.exp(0.5 * self.b_log_var)
        
        logits = F.linear(x, w.T, b)

        if return_var:
            # Monte Carlo estimate of predictive variance
            sample_logits = []
            for _ in range(10): # Number of samples for variance estimation
                w_sample = self.w_mean + torch.randn_like(self.w_mean) * torch.exp(0.5 * self.w_log_var)
                b_sample = self.b_mean + torch.randn_like(self.b_mean) * torch.exp(0.5 * self.b_log_var)
                sample_logits.append(F.linear(x, w_sample.T, b_sample))
            
            sample_probs = torch.sigmoid(torch.stack(sample_logits))
            variance = sample_probs.var(dim=0)
            return logits, variance
        
        return logits
    
    def _calculate_nll(self, logits: torch.Tensor, labels: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Numerically stable negative log-likelihood using log-sum-exp trick."""
        logsumexp = torch.logsumexp(logits, dim=1)
        chosen_logit = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        nll_per_sample = (logsumexp - chosen_logit) * weight
        return nll_per_sample.mean()

    def calculate_elbo(self, x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Calculates the Evidence Lower Bound (ELBO) for optimizing head parameters."""
        logits = self(x)
        
        # Expected log-likelihood term
        log_likelihood = -self._calculate_nll(logits, y, weight)
        
        # KL divergence between q(theta) and prior p(theta) ~ N(0,I)
        kl_w = 0.5 * torch.sum(torch.exp(self.w_log_var) + self.w_mean**2 - 1.0 - self.w_log_var)
        kl_b = 0.5 * torch.sum(torch.exp(self.b_log_var) + self.b_mean**2 - 1.0 - self.b_log_var)
        kl_divergence = kl_w + kl_b
        
        elbo = log_likelihood - kl_divergence
        # Return negative ELBO as the loss to be minimized
        return -elbo
    
    def calculate_margin_likelihood(self, x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor, sample_num: int = 5) -> torch.Tensor:
        """Approximates the marginal likelihood via Monte Carlo sampling for optimizing the shared bond."""
        sample_logits = [self(x) for _ in range(sample_num)]
        mean_logits = torch.mean(torch.stack(sample_logits), dim=0)
        
        # The loss for the bond is the negative log of the marginal likelihood
        return self._calculate_nll(mean_logits, y, weight)

class MT_DNN(torch.nn.Module):
    """The Multi-Task DNN architecture for the Margin Likelihood method."""
    def __init__(self, in_fdim: int, task_ids: list, layer_size: str, num_tasks: int):
        super(MT_DNN, self).__init__()
        self.in_fdim = in_fdim
        self.layer_size = layer_size
        self.num_tasks = num_tasks
        self.tasks_id_maps_heads = {task_id: idx for idx, task_id in enumerate(task_ids)}

        last_fdim = self.create_bond()
        self.create_sphead(last_fdim)

    def create_bond(self):
        """Creates the shared feature extractor (bond) layers."""
        # This logic is identical to other models and could be refactored into a common factory
        activation = nn.ReLU()
        dropout = nn.Dropout(p=0.5)     
        last_fdim = 1000
        ffn = []
        if self.layer_size == 'moderate':
            ffn.extend([nn.Linear(self.in_fdim, 1500), activation, dropout, nn.Linear(1500, 1000), activation])
        else:
             raise ValueError("Currently only 'moderate' layer_size is fully supported in this context.")
        self.bond = nn.Sequential(*ffn)
        return last_fdim
    
    def create_sphead(self, last_fdim: int):
        """Creates task-specific heads using the custom Variation_Layer."""
        self.heads = nn.ModuleList([Variation_Layer(last_fdim, 2) for _ in range(self.num_tasks)])
    
    def forward(self, in_feat: torch.Tensor, return_var: bool = False):
        """Forward pass through the shared bond and all task heads."""
        shared_features = self.bond(in_feat)
        return [head(shared_features, return_var=return_var) for head in self.heads]