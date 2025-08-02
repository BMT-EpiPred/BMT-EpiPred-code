# methods/Graph/MPNN_main.py

"""
Main script for training and evaluating the Multi-task MPNN model.

This script orchestrates the data loading for graph-based representations,
model training, evaluation across multiple tasks, and logging of performance metrics.

To run this script correctly from the project root directory, use the command:
python -m methods.Graph.MPNN_main
"""

import warnings
from pathlib import Path

import numpy as np
import torch

# Use relative import for the model controller
from .MPNN_mtask_model import Mtask_model
# Imports from root-level modules
from dataset import TrainGraphDataset, MPNNTestSmilesDataset
from config import args
from torch_geometric.loader import DataLoader as GraphDataLoader
from torch.utils.data import DataLoader as StandardDataLoader

def log_results(log_path, epoch, metrics, best_metrics):
    """Helper function to log metrics to a file."""
    with open(log_path, "a") as f:
        f.write(f"{'*'*20} epoch: {epoch} {'*'*20}\n")
        
        current_line = "valid set: " + "\t".join([f"{key.upper()} = {val}" for key, val in metrics.items()])
        best_line = "best:      " + "\t".join([f"{key.upper()} = {val}" for key, val in best_metrics.items()])
        
        f.write(current_line + "\n")
        f.write(best_line + "\n\n")

def main(split_idx: int, nn_heads: bool = False):
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

    param_dir = Path(f"MPNN_params/{args.dataset}_param/")
    result_dir = Path(f"MPNN_result/{args.dataset}_res/")
    param_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    model_save_path = param_dir / f'parameter_train_{train_percent}_{split_idx}.pkl'
    result_log_path = result_dir / f'{args.dataset}_train_{train_percent}_{split_idx}_valid.txt'
    ood_log_path = result_dir / f'{args.dataset}_train_{train_percent}_{split_idx}_ood.txt'

    # --- 3. Data Loading ---
    penalty_path = dataset_root / f"train_{train_percent}_{split_idx}_penalty.txt"
    with open(penalty_path, "r") as f:
        args.penalty_coefficients = list(map(float, f.readline().split()))
    
    # Use a dictionary for efficient O(1) average time complexity lookup
    smiles_to_data = {}
    for task_idx in range(args.num_tasks):
        task_id = args.task_no[task_idx]
        train_file = path_head / str(task_id) / f'train_{train_percent}_{split_idx}.txt'
        
        with open(train_file, 'r') as f:
            for line in f:
                smiles, label_str = line.split()
                if not smiles: continue
                label = int(label_str)

                if smiles not in smiles_to_data:
                    smiles_to_data[smiles] = {
                        "labels": np.zeros(args.num_tasks, dtype=int),
                        "weights": np.zeros(args.num_tasks, dtype=int)
                    }
                smiles_to_data[smiles]["labels"][task_idx] = label
                smiles_to_data[smiles]["weights"][task_idx] = 1

    s_train = list(smiles_to_data.keys())
    l_train = [data["labels"] for data in smiles_to_data.values()]
    weights = [data["weights"] for data in smiles_to_data.values()]

    # For MPNN, use the specific Graph-based datasets
    train_dataset = TrainGraphDataset(s_train, l_train, weights)
    # The DataLoader from torch_geometric is required for batching graph data
    train_loader = GraphDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Prepare validation data loaders
    valid_data_list, valid_data_taskID = [], []
    for task_id in args.task_no:
        valid_file = path_head / str(task_id) / f'valid_{train_percent}_{split_idx}.txt'
        s_valid, l_valid = [], []
        with open(valid_file, 'r') as f:
            for line in f:
                smiles, label = line.split()
                s_valid.append(smiles)
                l_valid.append(int(label))
        
        # This dataset returns SMILES, so a standard DataLoader is appropriate
        valid_dataset = MPNNTestSmilesDataset(s_valid, l_valid, task_id)
        valid_data_list.append(StandardDataLoader(valid_dataset, batch_size=args.batch_size_eval))
        valid_data_taskID.append(valid_dataset.get_task_id())

    # --- 4. Model Initialization & Metrics Tracking ---
    model = Mtask_model(args, NN_heads=nn_heads).to(device)
    
    metric_keys = ['acc', 'auc', 'aupr', 'f1', 'ba', 'ece', 'nll', 'ood_auroc']
    best_metrics = {key: 0.0 for key in metric_keys}

    # --- 5. Training Loop ---
    max_epoch = 1000
    for epoch in range(max_epoch + 1):
        torch.cuda.empty_cache()
        model.train_epoch(train_loader)

        if epoch % 10 == 0:
            eval_results = model.test_for_comp(valid_data_list, valid_data_taskID)
            current_metrics = dict(zip(metric_keys, eval_results))

            if current_metrics['auc'] > best_metrics['auc']:
                best_metrics = current_metrics.copy()
                torch.save(model.state_dict(), model_save_path)

            # Log results to files
            log_results(result_log_path, epoch, current_metrics, best_metrics)
            with open(ood_log_path, "a") as f:
                ood_info = {
                    'ood_auroc': current_metrics['ood_auroc'],
                    'acc': current_metrics['acc']
                }
                log_items = [f"epoch: {epoch}"] + [f"{key}: {val}" for key, val in ood_info.items()]
                f.write(', '.join(log_items) + '\n')

if __name__ == "__main__":
    # Loop through all data splits to run the full experiment
    for split_idx in [0, 1, 2]:
        print(f"\n===== Starting MPNN training for split index: {split_idx} =====\n")
        main(split_idx)
        print(f"\n===== Finished MPNN training for split index: {split_idx} =====\n")