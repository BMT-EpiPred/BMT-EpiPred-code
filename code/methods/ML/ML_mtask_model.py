# methods/ML/ML_mtask_model.py

"""
Controller class for the Multi-task model trained with Margin Likelihood (ML).

This module implements the unique alternating optimization training scheme:
1.  The shared encoder ('bond') is trained by maximizing the marginal likelihood.
2.  The task-specific heads are trained by maximizing the Evidence Lower Bound (ELBO).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn import metrics
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchmetrics.functional import calibration_error

# Use relative import for the model architecture
from .ML_model import MT_DNN

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Mtask_model(nn.Module):
    """Wrapper class for the alternating optimization training of the ML model."""
    def __init__(self, args):
        super(Mtask_model, self).__init__()
        self.args = args
        self.mtask_model = MT_DNN(args.emb_dim, args.task_no, args.layer_size, args.num_tasks)
        
        # Separate optimizers for the alternating training scheme
        self.bond_opt = optim.Adam(self.mtask_model.bond.parameters(), lr=args.lr, weight_decay=args.decay)
        self.head_opt = optim.Adam(self.mtask_model.heads.parameters(), lr=args.lr, weight_decay=args.decay)

        self.softmax = nn.Softmax(dim=1)

    def predict(self, init_feat: torch.Tensor, return_var: bool = False):
        """Generates predictions, optionally with variance."""
        outputs = self.mtask_model(init_feat, return_var=return_var)
        
        if return_var:
            logits, variances = zip(*outputs) # Unzip list of tuples
            probs = [self.softmax(logit)[:, 1] for logit in logits]
            return probs, variances
        else: # outputs is a list of logits
            probs = [self.softmax(logit)[:, 1] for logit in outputs]
            return probs

    def _train_bond_step(self, train_loader):
        """One pass through the data to train the shared encoder (bond)."""
        self.mtask_model.bond.train()
        self.mtask_model.heads.eval()
        
        for batch, weights in tqdm.tqdm(train_loader, desc=f"Training Bond {self.args.split_idx}", leave=False):
            features, y_true, weights = batch["vec"].to(device), batch["labels"].to(device, dtype=torch.long), weights.to(device)
            
            task_losses = []
            for i in range(self.args.num_tasks):
                loss = self.mtask_model.heads[i].calculate_margin_likelihood(
                    self.mtask_model.bond(features), y_true[:, i], weights[:, i]
                )
                task_losses.append(loss)
            
            total_loss = torch.stack(task_losses).mean()
            
            self.bond_opt.zero_grad()
            total_loss.backward()
            self.bond_opt.step()

    def _train_heads_step(self, train_loader):
        """One pass through the data to train the task-specific heads."""
        self.mtask_model.bond.eval()
        self.mtask_model.heads.train()

        # Freeze the bond parameters
        with torch.no_grad():
            shared_features = [self.mtask_model.bond(batch["vec"].to(device)) for batch, _ in train_loader]
            y_trues = [batch["labels"].to(device, dtype=torch.long) for batch, _ in train_loader]
            weights_list = [weights.to(device) for _, weights in train_loader]

        for i in range(len(shared_features)):
            features, y_true, weights = shared_features[i], y_trues[i], weights_list[i]
            
            task_losses = []
            for task_idx, head in enumerate(self.mtask_model.heads):
                loss = head.calculate_elbo(features, y_true[:, task_idx], weights[:, task_idx])
                task_losses.append(loss)
            
            total_loss = torch.stack(task_losses).mean()

            self.head_opt.zero_grad()
            total_loss.backward()
            self.head_opt.step()

    def train_epoch(self, train_loader):
        """Performs one full epoch of alternating optimization."""
        self._train_bond_step(train_loader)
        self._train_heads_step(train_loader)

    def _gather_predictions(self, data_list, task_id_list, with_uncertainty=False):
        """Internal helper to gather predictions and true labels."""
        self.mtask_model.eval()
        
        all_true, all_scores, all_vars = [], [], []

        with torch.no_grad():
            for task_idx, test_data in enumerate(data_list):
                y_true_task, y_scores_task, y_vars_task = [], [], []

                for batch in test_data:
                    features, labels = batch["vec"].to(device), batch["label"].to(device)
                    
                    if with_uncertainty:
                        probs, variances = self.predict(features, return_var=True)
                        y_vars_task.append(variances[task_idx])
                    else:
                        probs = self.predict(features)
                    
                    y_scores_task.append(probs[task_idx])
                    y_true_task.append(labels.view_as(probs[task_idx]))

                all_true.append(torch.cat(y_true_task).cpu())
                all_scores.append(torch.cat(y_scores_task).cpu())
                if with_uncertainty:
                    all_vars.append(torch.cat(y_vars_task).cpu())

        return all_true, all_scores, all_vars

    def test_for_comp(self, test_data_list, test_data_taskID):
        """Calculates and returns aggregated performance metrics."""
        y_true_list, y_scores_list, y_vars_list = self._gather_predictions(
            test_data_list, test_data_taskID, with_uncertainty=True
        )

        y_true = torch.cat(y_true_list).numpy()
        y_scores = torch.cat(y_scores_list).numpy()
        y_predict = (y_scores > 0.5).astype(int)
        
        # Placeholder for OOD, as this model doesn't have a built-in OOD score
        ood_auroc = 0.5 

        # Calculate metrics
        auc = metrics.roc_auc_score(y_true, y_scores)
        acc = metrics.accuracy_score(y_true, y_predict)
        aupr = metrics.average_precision_score(y_true, y_scores)
        f1 = metrics.f1_score(y_true, y_predict)
        ba = metrics.balanced_accuracy_score(y_true, y_predict)
        
        ece = calibration_error(torch.tensor(y_scores), torch.tensor(y_true), n_bins=15, task='binary').item()
        nll = F.binary_cross_entropy(torch.tensor(y_scores), torch.tensor(y_true, dtype=torch.float32)).item()
        
        return auc, acc, aupr, f1, ba, ece, nll, ood_auroc