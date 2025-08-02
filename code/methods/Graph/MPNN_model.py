# methods/Graph/MPNN_model.py

"""
Defines the core Multi-Task MPNN (Message Passing Neural Network) architecture.

This module wraps the MPNEncoder from the 'chemprop' library to serve as the
shared feature extractor ('bond') and combines it with task-specific prediction
layers ('heads'), which can be either standard NNs or VBLL layers.
"""

from argparse import Namespace

import torch
import torch.nn as nn

# Imports from root-level modules or installed packages
import VBLL_model
from chemprop.models.mpn import MPNEncoder
from chemprop.features.featurization import get_atom_fdim, get_bond_fdim

def _create_chemprop_args() -> Namespace:
    """Creates a Namespace object with default arguments required by chemprop's MPNEncoder."""
    # These arguments are based on the original script's setup.
    # It's good practice to centralize them.
    args = Namespace()
    args.hidden_size = 300
    args.bias = True
    args.depth = 3
    args.dropout = 0.0
    args.undirected = False
    args.atom_messages = True  # Using atom messages as per original implementation
    args.features_only = False
    args.use_input_features = False
    args.activation = 'ReLU'
    # These are required by the MPNEncoder's forward pass
    args.atom_fdim = get_atom_fdim()
    args.bond_fdim = get_bond_fdim()
    return args

class MT_DNN(torch.nn.Module):
    """
    Multi-Task Deep Neural Network using an MPNN encoder.
    """
    def __init__(self, task_ids: list, layer_size: str, num_tasks: int,
                 regularization_weight: float,
                 parameterization: str = 'dense',
                 prior_scale: float = 1.0,
                 NN_heads: bool = False,
                 **kwargs):
        super(MT_DNN, self).__init__()
        
        self.num_tasks = num_tasks
        self.tasks_id_maps_heads = {task_id: idx for idx, task_id in enumerate(task_ids)}

        # Create the shared MPNN encoder
        last_fdim = self.create_bond()
        
        # Create the task-specific prediction heads
        self.create_sphead(
            last_fdim=last_fdim,
            regularization_weight=regularization_weight,
            parameterization=parameterization,
            prior_scale=prior_scale,
            NN_heads=NN_heads,
            **kwargs  # Pass any remaining VBLL params
        )

    def create_bond(self) -> int:
        """
        Initializes the shared MPNEncoder from chemprop.
        """
        self.chemprop_args = _create_chemprop_args()
        self.bond = MPNEncoder(self.chemprop_args)
        return self.chemprop_args.hidden_size
    
    def create_sphead(self, last_fdim: int, regularization_weight: float,
                      parameterization: str, prior_scale: float,
                      NN_heads: bool, **kwargs):
        """Creates the task-specific output layers (heads)."""
        if self.num_tasks < 1:
            raise ValueError("Number of tasks must be at least 1.")
        
        heads = []
        for _ in range(self.num_tasks):
            if NN_heads:
                head = nn.Linear(last_fdim, 2)
            else:
                head = VBLL_model.VBLL_Layer(
                    in_features=last_fdim,
                    out_features=2,
                    regularization_weight=regularization_weight,
                    parameterization=parameterization,
                    prior_scale=prior_scale,
                    **kwargs
                )
            heads.append(head)
        
        self.heads = nn.ModuleList(heads)
    
    def forward(self, smiles_batch: list) -> list:
        """
        Performs a forward pass through the MPNN and all task heads.
        
        Args:
            smiles_batch: A list of SMILES strings.
        
        Returns:
            A list of outputs, one for each task head.
        """
        # The MPNEncoder from chemprop handles the conversion from SMILES to graph internally
        shared_representation = self.bond(smiles_batch)
   
        # Compute output for each task through its corresponding head
        task_outputs = [head(shared_representation) for head in self.heads]
        
        return task_outputs