# methods/ML/ML_main.py

"""
Main script for training and evaluating the Multi-task model using the
Margin Likelihood (ML) alternating optimization strategy.
"""

from pathlib import Path
import torch
import numpy as np

# Use relative import for the model controller
from .ML_mtask_model import Mtask_model
# Imports from root-level modules
from dataset import ECFPDataset, SmilesDataset
from torch.utils.data import DataLoader
from config import args

def log_results(log_path, epoch, metrics, best_metrics):
    """Helper function to log metrics to a file."""
    with open(log_path, "a") as f:
        f.write(f"{'*'*20} epoch: {epoch} {'*'*20}\n")
        
        metrics_subset = {k: v for k, v in metrics.items() if k not in ['mcc', 'precision', 'recall']}
        best_metrics_subset = {k: v for k, v in best_metrics.items() if k not in ['mcc', 'precision', 'recall']}

        current_line = "valid set: " + "\t".join([f"{key.upper()} = {val}" for key, val in metrics_subset.items()])
        best_line = "best:      " + "\t".join([f"{key.upper()} = {val}" for key, val in best_metrics_subset.items()])
        
        f.write(current_line + "\n")
        f.write(best_line + "\n\n")

def main(split_idx: int):
    """Main training and evaluation function for a single data split."""
    args.split_idx = split_idx

    # --- 1. Environment Setup ---
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    # --- 2. Path Configuration ---
    train_percent = 80
    path_head = Path(args.path_head)
    dataset_root = path_head.parent

    param_dir = Path(f"params/{args.dataset}_param/")
    result_dir = Path(f"result/{args.dataset}_res/")
    param_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    model_save_path = param_dir / f'parameter_train_{train_percent}_{split_idx}_best_valid_auc.pkl'
    result_log_path = result_dir / f'{args.dataset}_train_{train_percent}_{split_idx}_valid.txt'

    # --- 3. Data Loading ---
    penalty_path = dataset_root / f"train_{train_percent}_{split_idx}_penalty.txt"
    with open(penalty_path, "r") as f:
        args.penalty_coefficients = list(map(float, f.readline().split()))
    
    smiles_to_data = {}
    for task_idx in range(args.num_tasks):
        task_id = args.task_no[task_idx]
        train_file = path_head / str(task_id) / f'train_{train_percent}_{split_idx}.txt'
        with open(train_file, 'r') as f:
            for line in f:
                smiles, label_str = line.split()
                if not smiles: continue
                if smiles not in smiles_to_data:
                    smiles_to_data[smiles] = {
                        "labels": np.zeros(args.num_tasks, dtype=int),
                        "weights": np.zeros(args.num_tasks, dtype=int)
                    }
                smiles_to_data[smiles]["labels"][task_idx] = int(label_str)
                smiles_to_data[smiles]["weights"][task_idx] = 1

    s_train = list(smiles_to_data.keys())
    l_train = [data["labels"] for data in smiles_to_data.values()]
    weights = [data["weights"] for data in smiles_to_data.values()]
    
    train_loader = DataLoader(SmilesDataset(s_train, l_train, weights), batch_size=args.batch_size, shuffle=True)

    valid_data_list, valid_data_taskID = [], []
    for task_id in args.task_no:
        valid_file = path_head / str(task_id) / f'valid_{train_percent}_{split_idx}.txt'
        s_valid, l_valid = [], []
        with open(valid_file, 'r') as f:
            for line in f:
                smiles, label = line.split()
                s_valid.append(smiles)
                l_valid.append(int(label))
        
        valid_dataset = ECFPDataset(s_valid, l_valid, task_id)
        valid_data_list.append(DataLoader(valid_dataset, batch_size=args.batch_size_eval))
        valid_data_taskID.append(valid_dataset.get_task_id())

    # --- 4. Model Initialization & Training ---
    model = Mtask_model(args).to(device)
    
    metric_keys = ['auc', 'acc', 'aupr', 'f1', 'ba', 'ece', 'nll', 'ood_auroc']
    best_metrics = {key: 0.0 for key in metric_keys}

    for epoch in range(args.epoch + 1):
        torch.cuda.empty_cache()
        model.train_epoch(train_loader)

        if epoch % 10 == 0:
            eval_results = model.test_for_comp(valid_data_list, valid_data_taskID)
            current_metrics = dict(zip(metric_keys, eval_results))

            if current_metrics['auc'] > best_metrics['auc']:
                best_metrics = current_metrics.copy()
                torch.save(model.state_dict(), model_save_path)

            log_results(result_log_path, epoch, current_metrics, best_metrics)

if __name__ == "__main__":
    for split_idx in [0, 1, 2]:
        print(f"\n===== Starting ML training for split index: {split_idx} =====\n")
        main(split_idx)
        print(f"\n===== Finished ML training for split index: {split_idx} =====\n")