# methods/ECFP/main.py

"""
Main script for training and evaluating the Multi-task model using ECFP features.

This script handles data loading, model training, evaluation, and logging of results
for a specified number of data splits.

To run this script correctly from the project root directory, use the following command:
python -m methods.ECFP.main
"""

import os
import warnings
from pathlib import Path
import torch
import numpy as np
import pandas as pd

from methods.ECFP.ECFP_mtask_model import Mtask_model
from dataset import ECFPDataset, SmilesDataset
from torch.utils.data import DataLoader
from config import args

def log_results(log_path, epoch, metrics, best_metrics, is_full_data_log=False):
    """Helper function to log metrics to a file."""
    # Use 'a' mode to append to the file
    with open(log_path, "a") as f:
        if is_full_data_log:
            # Log for plotting full data over epochs
            log_items = [f"epoch: {epoch}"] + [f"{key}: {val}" for key, val in metrics.items()]
            f.write(', '.join(log_items) + '\n')
        else:
            # Log for standard validation results
            f.write(f"{'*'*20} epoch: {epoch} {'*'*20}\n")
            
            current_line = "valid set: " + "\t".join([f"{key.upper()} = {val}" for key, val in metrics.items()])
            best_line = "best:      " + "\t".join([f"{key.upper()} = {val}" for key, val in best_metrics.items()])
            
            f.write(current_line + "\n")
            f.write(best_line + "\n\n")

def main(split_idx: int):
    """Main training and evaluation function for a single data split."""
    warnings.filterwarnings("ignore", message="DEPRECATION WARNING: please use MorganGenerator")
    args.split_idx = split_idx

    # --- 1. Setup Environment ---
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    # --- 2. Configure Paths ---
    # Using pathlib for robust, OS-agnostic path handling.
    # Assumes the script is run from the project root directory.
    train_percent = 80
    path_head = Path(args.path_head)
    dataset_root = path_head.parent  # Gets the directory containing the 'split_X' folders

    # Define output directories relative to the project root
    param_dir = Path(f"params/{args.dataset}_param/")
    result_dir = Path(f"result/{args.dataset}_res/")
    param_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    model_save_path = param_dir / f'parameter_train_{train_percent}_{split_idx}_best_valid_auc.pkl'
    result_log_path = result_dir / f'{args.dataset}_train_{train_percent}_{split_idx}_valid.txt'
    full_data_log_path = result_dir / f'{args.dataset}_train_{train_percent}_{split_idx}_fulldata.txt'

    # --- 3. Load Data ---
    # Load penalty coefficients for data balancing
    penalty_path = dataset_root / f"train_{train_percent}_{split_idx}_penalty.txt"
    with open(penalty_path, "r") as f:
        args.penalty_coefficients = list(map(float, f.readline().split()))
    
    # Use a dictionary for efficient SMILES lookup (O(1) average time complexity)
    # This avoids the O(N^2) complexity of list.index() in a loop.
    smiles_to_data = {}

    for task_idx in range(args.num_tasks):
        task_id = args.task_no[task_idx]
        train_file = path_head / str(task_id) / f'train_{train_percent}_{split_idx}.txt'
        
        with open(train_file, 'r') as f:
            for line in f:
                smiles, label_str = line.split()
                label = int(label_str)

                if smiles not in smiles_to_data:
                    # Initialize labels and weights for a new molecule
                    labels = np.zeros(args.num_tasks, dtype=int)
                    weights = np.zeros(args.num_tasks, dtype=int)
                    smiles_to_data[smiles] = [labels, weights]

                # Update the specific task's label and weight
                smiles_to_data[smiles][0][task_idx] = label
                smiles_to_data[smiles][1][task_idx] = 1

    # Unpack the dictionary into lists for the DataLoader
    s_train = list(smiles_to_data.keys())
    l_train = [data[0] for data in smiles_to_data.values()]
    weights = [data[1] for data in smiles_to_data.values()]
    
    train_loader = DataLoader(
        SmilesDataset(s_train, l_train, weights),
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=True
    )

    # Prepare validation data loaders
    valid_data_list = []
    valid_data_taskID = []
    for task_id in args.task_no:
        valid_file = path_head / str(task_id) / f'valid_{train_percent}_{split_idx}.txt'
        s_valid, l_valid = [], []
        with open(valid_file, 'r') as f:
            for line in f:
                smiles, label = line.split()
                s_valid.append(smiles)
                l_valid.append(int(label))
        
        task_valid_set = ECFPDataset(s_valid, l_valid, task_id)
        valid_data_list.append(DataLoader(task_valid_set, batch_size=args.batch_size_eval))
        valid_data_taskID.append(task_valid_set.get_task_id())

    # --- 4. Initialize Model and Metrics ---
    model = Mtask_model(args, s_train).to(device)
    
    # Use a dictionary to track best metrics for cleaner code
    metric_keys = ['acc', 'auc', 'aupr', 'f1', 'ba', 'ece', 'nll', 'ood_auroc', 'mcc', 'precision', 'recall']
    best_metrics = {key: 0.0 for key in metric_keys}

    # --- 5. Training Loop ---
    for epoch in range(args.epoch + 1):
        torch.cuda.empty_cache()
        model.train(train_loader)

        if epoch % 10 == 0:
            # Evaluate the model on the validation set
            eval_results = model.test_for_comp(valid_data_list, valid_data_taskID)
            
            # Unpack results into a dictionary for easy handling
            current_metrics = dict(zip(metric_keys, eval_results))

            # Check for improvement and save the best model
            if current_metrics['auc'] > best_metrics['auc']:
                best_metrics = current_metrics.copy()
                torch.save(model.state_dict(), model_save_path)

            # Log results using the helper function
            log_results(result_log_path, epoch, current_metrics, best_metrics)
            log_results(full_data_log_path, epoch, current_metrics, None, is_full_data_log=True)

if __name__ == "__main__":
    # Loop through all data splits to run the full experiment
    for split_idx in [0, 1, 2]:
        print(f"\n===== Starting training for split index: {split_idx} =====\n")
        main(split_idx)
        print(f"\n===== Finished training for split index: {split_idx} =====\n")