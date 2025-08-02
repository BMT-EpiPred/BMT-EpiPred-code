# methods/ECFP/ECFP_model.py

"""
Defines the core Multi-Task Deep Neural Network (MT-DNN) architecture.

This module contains the MT_DNN class, which dynamically constructs a shared
feature extractor ('bond') and task-specific prediction layers ('heads').
It supports different network depths and can use either standard neural network
heads or Variational Bayesian Last Layer (VBLL) heads for uncertainty estimation.
"""

import torch
from torch import nn
import VBLL_model  # Assumes VBLL_model.py is in the project root

class MT_DNN(torch.nn.Module):
    """
    Multi-Task Deep Neural Network.
    
    This network consists of a shared 'bond' of fully-connected layers and multiple
    task-specific 'heads' for predictions.
    """
    def __init__(self, in_fdim, task_ids, layer_size, num_tasks,
                 regularization_weight,
                 parameterization='dense',
                 softmax_bound='jensen',
                 return_ood=False,
                 prior_scale=1.,
                 wishart_scale=0.1, # Adjusted from original for potential better performance
                 dof=1.,
                 cov_rank=3,
                 NN_heads=False):
        super(MT_DNN, self).__init__()
        self.in_fdim = in_fdim
        self.layer_size = layer_size
        self.num_tasks = len(task_ids)
        
        # Maps a task ID to its corresponding head index
        self.tasks_id_maps_heads = {task_id: idx for idx, task_id in enumerate(task_ids)}

        # Create the shared network layers and the task-specific heads
        last_fdim = self.create_bond()
        self.create_sphead(last_fdim,
                           regularization_weight,
                           parameterization,
                           softmax_bound,
                           return_ood,
                           prior_scale,
                           wishart_scale,
                           dof,
                           cov_rank,
                           NN_heads=NN_heads)

    def create_bond(self):
        """
        Creates the shared feature extractor (the 'bond') of the network.
        This part is common across all tasks.
        """
        activation = nn.ReLU()
        dropout = nn.Dropout(p=0.5)
        last_fdim = 1000  # Default output dimension
        ffn = []

        if self.layer_size == 'shallow':
            # [1000]
            ffn.extend([nn.Linear(self.in_fdim, 1000), activation])
        
        elif self.layer_size == 'moderate':
            # [1500, 1000]
            ffn.extend([nn.Linear(self.in_fdim, 1500), activation])
            ffn.extend([dropout, nn.Linear(1500, 1000), activation])
            
        elif self.layer_size == 'deep':
            # [2000, 1000, 500]
            ffn = [nn.Linear(self.in_fdim, 2000), activation]
            ffn.extend([dropout, nn.Linear(2000, 1000), activation])
            ffn.extend([dropout, nn.Linear(1000, 500), activation])
            last_fdim = 500
        
        elif self.layer_size == 'task_relate':
            # [1024, num_tasks]
            ffn.extend([nn.Linear(self.in_fdim, 1024), activation])
            ffn.extend([dropout, nn.Linear(1024, self.num_tasks), activation])
            last_fdim = self.num_tasks
        
        else:
             raise ValueError("Unsupported layer_size. Choose from 'shallow', 'moderate', 'deep', 'task_relate'.")
        
        self.bond = nn.Sequential(*ffn)
        return last_fdim
    
    def create_sphead(self, last_fdim, regularization_weight, parameterization,
                      softmax_bound, return_ood, prior_scale, wishart_scale,
                      dof, cov_rank, NN_heads=False):
        """Creates the task-specific output layers (the 'heads')."""
        if self.num_tasks < 1:
            raise ValueError("Number of tasks must be at least 1.")
        
        heads = []
        for _ in range(self.num_tasks):
            # Each task has its own specific output layer (head)
            if NN_heads:
                # Standard deterministic feed-forward layer
                head = nn.Linear(last_fdim, 2)
            else:
                # Variational Bayesian Last Layer for uncertainty
                head = VBLL_model.VBLL_Layer(
                    in_features=last_fdim,
                    out_features=2,
                    regularization_weight=regularization_weight,
                    parameterization=parameterization,
                    softmax_bound=softmax_bound,
                    return_ood=return_ood,
                    prior_scale=prior_scale,
                    wishart_scale=wishart_scale,
                    dof=dof,
                    cov_rank=cov_rank
                )
            heads.append(head)
        
        self.heads = nn.ModuleList(heads)
    
    def forward(self, in_feat):
        """Forward pass through the network."""
        # Get shared representation
        shared_representation = self.bond(in_feat)
   
        # Compute output for each task through its corresponding head
        task_outputs = [head(shared_representation) for head in self.heads]
        
        return task_outputs