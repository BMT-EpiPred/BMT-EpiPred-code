# methods/ECFP/ECFP_mtask_model.py

"""
Controller class for the Multi-task ECFP model.

This module wraps the MT_DNN model, managing the training loop, evaluation,
prediction, and metric calculation. It handles both standard NN heads and
VBLL heads for uncertainty-aware predictions.

To run scripts that use this model, execute from the project root directory,
e.g., `python -m methods.ECFP.ECFP_main`.
"""

import os
from typing import List
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
from torchmetrics.functional import calibration_error

# Use relative import for the model architecture, which is robust
from .ECFP_model import MT_DNN
# Imports from root-level modules
import VBLL_model
from dataset import OODDataset # Assuming OODDataset handles ECFP features

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_ood_data():
    """Helper function to load and prepare OOD datasets."""
    # Note: Hardcoded paths are not ideal. Consider moving to config.
    # Note: This loads and processes data on every call, which is inefficient.
    # For real use, this data should be pre-processed and cached.
    try:
        davis_path = Path('./dataset/Davis/davis-filter-smiles.txt')
        with open(davis_path, 'r') as f:
            davis_smi = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Warning: Davis dataset not found at {davis_path}. OOD evaluation will be impacted.")
        davis_smi = []
    
    qm9_val_size = 34000 - len(davis_smi)
    qm9_smi = []
    try:
        from torch_geometric.datasets import QM9
        qm9_path = Path('./dataset/QM9')
        qm9_dataset = QM9(str(qm9_path)).shuffle()
        for qm9_data in tqdm.tqdm(qm9_dataset, desc="Processing QM9 for OOD", leave=False):
            if Chem.MolFromSmiles(qm9_data.smiles):
                qm9_smi.append(qm9_data.smiles)
                if len(qm9_smi) >= qm9_val_size:
                    break
    except (ImportError, FileNotFoundError):
        print(f"Warning: QM9 dataset not found or torch_geometric not installed. OOD evaluation will be impacted.")
    
    return davis_smi + qm9_smi

class Mtask_model(nn.Module):
    """
    Wrapper class for training and evaluating the multi-task model.
    """
    def __init__(self, args, s_train, NN_heads=False):
        super(Mtask_model, self).__init__()
    
        self.args = args
        self.NN_heads = NN_heads
        
        # Initialize the underlying MT-DNN model
        self.mtask_model = MT_DNN(
            in_fdim=args.emb_dim,
            task_ids=args.task_no,
            layer_size=args.layer_size,
            num_tasks=args.num_tasks,
            regularization_weight=1./len(s_train),
            parameterization='diagonal',
            return_ood=True,
            prior_scale=1.0,
            NN_heads=NN_heads
        )

        # Setup optimizer with different weight decay for bond and heads
        if not self.NN_heads:
            model_param_group = [
                {'params': self.mtask_model.bond.parameters(), 'weight_decay': args.decay},
                {'params': self.mtask_model.heads.parameters(), 'weight_decay': 0.}
            ]
            self.opt = optim.AdamW(model_param_group, lr=args.lr)
        else:
            self.opt = optim.Adam(self.mtask_model.parameters(), lr=args.lr, weight_decay=args.decay)

        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input_features):
        return self.mtask_model(input_features)
    
    def predict(self, init_feat, return_uncertainty=False):
        """
        Makes predictions for all tasks on a given input batch.
        Can optionally return uncertainty estimates for VBLL models.
        """
        task_outputs = self.forward(init_feat)
        
        tasks_preds = [None] * len(task_outputs)
        tasks_uncertainty = [None] * len(task_outputs)

        if self.NN_heads:
            for i, out in enumerate(task_outputs):
                tasks_preds[i] = self.softmax(out)[:, 1]
        else:
            # VBLL heads
            hidden = self.mtask_model.bond(init_feat)
            for i, out in enumerate(task_outputs):
                tasks_preds[i] = out.predictive.probs[:, 1]
                if return_uncertainty:
                    # Simplified uncertainty calculation; assumes single head contribution
                    # A more complex model might aggregate uncertainty differently.
                    tasks_uncertainty[i] = out.predictive.scales[:, 1]

        return (tasks_preds, tasks_uncertainty) if return_uncertainty else tasks_preds

    def train_epoch(self, multi_task_train_data):
        self.mtask_model.train()

        for _, (batch, weights) in enumerate(tqdm.tqdm(multi_task_train_data, desc=f"{self.args.layer_size}-{self.args.split_idx}")):
            init_features = batch["vec"].to(device)
            y_true = batch['labels'].to(device, dtype=torch.long)
            weights = weights.to(device)
            
            outputs = self.forward(init_features)
            
            # Prepare penalty tensor outside the loop to avoid duplication
            penalty = torch.DoubleTensor(self.args.penalty_coefficients).to(device)
            penalty = penalty.repeat(y_true.size(0), 1).mul(y_true.to(torch.float))
            penalty = torch.where(penalty == 0, 1.0, penalty)

            # Calculate loss based on head type
            if self.NN_heads:
                outputs = torch.stack(outputs).transpose(0, 1) # Shape: [batch, tasks, classes]
                criterion = nn.CrossEntropyLoss(reduction='none')
                
                # Manual loop for NN heads to apply weights
                total_loss = 0
                for i in range(outputs.size(0)): # Iterate over batch
                    # Loss for all tasks for a single sample
                    sample_loss = criterion(outputs[i].double(), y_true[i])
                    # Apply task weights and data balancing penalties
                    weighted_loss = (sample_loss * weights[i] * penalty[i]).sum()
                    total_loss += weighted_loss
                total_loss /= outputs.size(0)

            else: # VBLL heads
                task_losses = []
                for i, out in enumerate(outputs):
                    # VBLL layer computes its own loss
                    loss = out.train_loss_fn(y_true[:, i])
                    # Apply task weights and data balancing penalties
                    weighted_loss = loss * weights[:, i] * penalty[:, i]
                    task_losses.append(weighted_loss)
                
                total_loss = torch.stack(task_losses).sum()

            self.opt.zero_grad()
            total_loss.backward()
            self.opt.step()

    def _gather_predictions(self, data_list, task_id_list, with_uncertainty=False):
        """Internal helper to gather predictions and true labels from dataloaders."""
        self.mtask_model.eval()
        
        all_true = []
        all_scores = []
        all_vars = [] if with_uncertainty else None

        with torch.no_grad():
            for task_idx, test_data in enumerate(data_list):
                task_id = task_id_list[task_idx]
                
                y_true_task, y_scores_task, y_vars_task = [], [], []

                for batch in tqdm.tqdm(test_data, desc=f"Evaluating Task {task_id}", leave=False):
                    features = batch["vec"].to(device)
                    labels = batch["label"].to(device)

                    if with_uncertainty:
                        preds, uncertainties = self.predict(features, return_uncertainty=True)
                        y_vars_task.append(uncertainties[task_idx])
                    else:
                        preds = self.predict(features, return_uncertainty=False)

                    y_scores_task.append(preds[task_idx])
                    y_true_task.append(labels.view_as(preds[task_idx]))
                
                all_true.append(torch.cat(y_true_task, dim=0).cpu())
                all_scores.append(torch.cat(y_scores_task, dim=0).cpu())
                if with_uncertainty:
                    all_vars.append(torch.cat(y_vars_task, dim=0).cpu())

        return all_true, all_scores, all_vars

    def eval_ood(self, ind_dataloader, ood_dataloader, task_idx):
        """Evaluates OOD detection performance for a single task."""
        self.mtask_model.eval()
        
        def get_scores(loader, is_ood=False):
            scores = []
            with torch.no_grad():
                for batch in loader:
                    x = batch["vec"].to(device)
                    out = self.mtask_model(x)[task_idx]
                    if self.NN_heads:
                        score = torch.max(F.softmax(out, dim=-1), dim=-1)[0]
                    else: # VBLL
                        score = out.ood_scores
                    scores.append(score.cpu())
            return torch.cat(scores).numpy()

        ind_scores = get_scores(ind_dataloader)
        ood_scores = get_scores(ood_dataloader)

        labels = np.concatenate([np.ones_like(ind_scores), np.zeros_like(ood_scores)])
        scores = np.concatenate([ind_scores, ood_scores])
        
        return metrics.roc_auc_score(labels, scores)

    def test_for_comp(self, test_data_list, test_data_taskID):
        """
        Calculates and returns aggregated performance metrics across all tasks.
        This is the main evaluation function.
        """
        y_true_list, y_scores_list, y_vars_list = self._gather_predictions(
            test_data_list, test_data_taskID, with_uncertainty=not self.NN_heads
        )

        # Concatenate results from all tasks for overall metrics
        y_true = torch.cat(y_true_list).numpy()
        y_scores = torch.cat(y_scores_list).numpy()
        y_predict = (y_scores > 0.5).astype(int)

        # OOD Evaluation (on a subset of tasks for efficiency)
        ood_smiles = _load_ood_data()
        ood_auroc_scores = []
        for task_idx in range(min(3, len(test_data_list))): # Limit to first 3 tasks
            ood_dataset = OODDataset(ood_smiles) # Assuming OODDataset creates ECFP vectors
            ood_dataloader = DataLoader(ood_dataset, batch_size=128)
            auroc = self.eval_ood(test_data_list[task_idx], ood_dataloader, task_idx)
            ood_auroc_scores.append(auroc)
        
        ood_auroc = np.mean(ood_auroc_scores) if ood_auroc_scores else 0.0

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

        return auc, acc, aupr, f1, ba, ece, nll, ood_auroc, mcc, precision, recall