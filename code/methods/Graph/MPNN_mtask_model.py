# methods/Graph/MPNN_mtask_model.py

"""
Controller class for the Multi-task MPNN model.

This module wraps the MPNN-based MT_DNN model, managing the training loop,
evaluation, prediction, and metric calculation for graph-based inputs.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
from rdkit import Chem
from sklearn import metrics
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GraphDataLoader
from torchmetrics.functional import calibration_error

# Use relative import for the model architecture
from .MPNN_model import MT_DNN
# Imports from root-level modules
from dataset import OODMPNNTestSmilesDataset

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Mtask_model(nn.Module):
    """
    Wrapper class for training and evaluating the multi-task MPNN model.
    """
    def __init__(self, args, NN_heads=False):
        super(Mtask_model, self).__init__()
    
        self.args = args
        self.NN_heads = NN_heads
        
        self.mtask_model = MT_DNN(
            task_ids=args.task_no,
            layer_size=args.layer_size,
            num_tasks=args.num_tasks,
            # Placeholder for regularization weight, as s_train is not passed.
            # This should be configured properly if using VBLL.
            regularization_weight=1e-5, 
            NN_heads=NN_heads
        )

        # Setup optimizer
        if not self.NN_heads:
            model_param_group = [
                {'params': self.mtask_model.bond.parameters(), 'weight_decay': args.decay},
                {'params': self.mtask_model.heads.parameters(), 'weight_decay': 0.}
            ]
            self.opt = optim.AdamW(model_param_group, lr=args.lr)
        else:
            self.opt = optim.Adam(self.mtask_model.parameters(), lr=args.lr, weight_decay=args.decay)

        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, smiles_batch):
        return self.mtask_model(smiles_batch)

    def predict(self, smiles_batch: list) -> list:
        """
        Makes predictions for all tasks on a given batch of SMILES strings.
        """
        task_outputs = self.forward(smiles_batch)
        
        tasks_preds = [None] * len(task_outputs)
        for i, out in enumerate(task_outputs):
            if self.NN_heads:
                tasks_preds[i] = self.softmax(out)[:, 1]
            else: # VBLL
                tasks_preds[i] = out.predictive.probs[:, 1]
        
        return tasks_preds

    def train_epoch(self, train_loader: GraphDataLoader):
        self.mtask_model.train()

        for batch in tqdm.tqdm(train_loader, desc=f"Training {self.args.layer_size}-{self.args.split_idx}"):
            # The chemprop MPNEncoder takes SMILES strings directly
            smiles = batch.smiles
            y_true = batch.y.to(device, dtype=torch.long)
            weights = batch.weight.to(device)
            
            outputs = self.forward(smiles)
            
            # Prepare penalty tensor
            penalty = torch.DoubleTensor(self.args.penalty_coefficients).to(device)
            penalty = penalty.repeat(y_true.size(0), 1).mul(y_true.to(torch.float))
            penalty = torch.where(penalty == 0, 1.0, penalty)

            # Calculate loss
            if self.NN_heads:
                outputs = torch.stack(outputs).transpose(0, 1) # Shape: [batch, tasks, classes]
                criterion = nn.CrossEntropyLoss(reduction='none')
                
                # Apply loss, weights, and penalty per sample
                task_losses = criterion(outputs.double(), y_true) # Shape: [batch, tasks]
                weighted_losses = (task_losses * weights * penalty).sum()
                total_loss = weighted_losses / y_true.size(0)

            else: # VBLL heads
                task_losses = []
                for i, out in enumerate(outputs):
                    loss = out.train_loss_fn(y_true[:, i])
                    weighted_loss = loss * weights[:, i] * penalty[:, i]
                    task_losses.append(weighted_loss)
                total_loss = torch.stack(task_losses).sum()

            self.opt.zero_grad()
            total_loss.backward()
            self.opt.step()

    def _gather_predictions(self, data_list, task_id_list):
        """Internal helper to gather predictions and true labels."""
        self.mtask_model.eval()
        
        all_true = []
        all_scores = []

        with torch.no_grad():
            for task_idx, test_data in enumerate(data_list):
                task_id = task_id_list[task_idx]
                
                y_true_task, y_scores_task = [], []

                for smiles, labels in tqdm.tqdm(test_data, desc=f"Evaluating Task {task_id}", leave=False):
                    labels = labels.to(device)
                    preds = self.predict(smiles)[task_idx]
                    
                    y_scores_task.append(preds)
                    y_true_task.append(labels.view_as(preds))
                
                all_true.append(torch.cat(y_true_task, dim=0).cpu())
                all_scores.append(torch.cat(y_scores_task, dim=0).cpu())

        return all_true, all_scores

    def test_for_comp(self, test_data_list, test_data_taskID):
        """
        Calculates and returns aggregated performance metrics across all tasks.
        """
        y_true_list, y_scores_list = self._gather_predictions(test_data_list, test_data_taskID)

        y_true = torch.cat(y_true_list).numpy()
        y_scores = torch.cat(y_scores_list).numpy()
        y_predict = (y_scores > 0.5).astype(int)

        # Note: OOD evaluation for MPNN would require a separate flow since it needs SMILES.
        # This part is simplified or can be adapted if an OOD dataset with SMILES is provided.
        ood_auroc = 0.5 # Placeholder value

        # Calculate metrics
        auc = metrics.roc_auc_score(y_true, y_scores)
        acc = metrics.accuracy_score(y_true, y_predict)
        aupr = metrics.average_precision_score(y_true, y_scores)
        f1 = metrics.f1_score(y_true, y_predict)
        ba = metrics.balanced_accuracy_score(y_true, y_predict)
        mcc = metrics.matthews_corrcoef(y_true, y_predict)
        precision = metrics.precision_score(y_true, y_predict)
        recall = metrics.recall_score(y_true, y_predict)
        
        ece = calibration_error(torch.tensor(y_scores), torch.tensor(y_true), n_bins=15, task='binary').item()
        nll = F.binary_cross_entropy(torch.tensor(y_scores), torch.tensor(y_true, dtype=torch.float32)).item()
        
        return acc, auc, aupr, f1, ba, ece, nll, ood_auroc