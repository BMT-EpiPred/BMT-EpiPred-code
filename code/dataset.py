# dataset.py

"""
PyTorch Dataset classes for handling molecular data.

This module provides a suite of Dataset objects for different molecular
representations (ECFP, Graph) and use cases (training, evaluation, OOD).
It leverages a featurizer design pattern to separate data representation
from the dataset logic itself.
"""
import warnings
import torch
from torch.utils.data import Dataset
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np
from torch_geometric.data import Data

# It's better practice to import these from a separate constants file.
from constants import FAMILIES_DICT, IDX_TO_NAME_DICT

# --- Base Featurizer Classes ---

class BaseFeaturizer:
    """Abstract base class for molecular featurizers."""
    def __call__(self, smiles: str):
        raise NotImplementedError

class ECFPFeaturizer(BaseFeaturizer):
    """Computes ECFP4 (Morgan) fingerprints for a given SMILES string."""
    def __init__(self, radius: int = 2, n_bits: int = 1024):
        self.radius = radius
        self.n_bits = n_bits

    def __call__(self, smiles: str) -> np.ndarray:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            warnings.warn(f"Could not parse SMILES: {smiles}. Returning a zero vector.")
            return np.zeros(self.n_bits, dtype=np.float32)
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
        arr = np.zeros((0,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

class GraphFeaturizer(BaseFeaturizer):
    """Constructs a graph representation from a SMILES string."""
    def _get_atom_features(self, mol):
        """Helper to extract atom features."""
        features = []
        for atom in mol.GetAtoms():
            features.append([
                atom.GetAtomicNum(),
                atom.GetFormalCharge(),
                int(atom.GetIsAromatic()),
                atom.GetMass(),
                int(atom.GetHybridization()),
                atom.GetTotalNumHs(),
            ])
        return np.array(features, dtype=np.float32)

    def __call__(self, smiles: str) -> Data:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            warnings.warn(f"Could not parse SMILES: {smiles}. Returning an empty graph.")
            return Data(x=torch.empty(0, 6), edge_index=torch.empty(2, 0, dtype=torch.long))

        # Atom (node) features
        atom_features = torch.tensor(self._get_atom_features(mol), dtype=torch.float32)
        
        # Bond (edge) features and connectivity
        edge_indices, edge_attrs = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices.append((i, j))
            edge_indices.append((j, i)) # Add both directions for undirected graph
            
            bond_type = [0.0] * 4 # e.g., single, double, triple, aromatic
            bond_type[int(bond.GetBondTypeAsDouble()) - 1] = 1.0
            edge_attrs.append(bond_type)
            edge_attrs.append(bond_type)
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)

        return Data(x=atom_features, edge_index=edge_index, edge_attr=edge_attr)


# --- Dataset Classes ---

class BaseDataset(Dataset):
    """Base class for datasets, holding common logic."""
    def __init__(self, smiles, labels=None):
        self.smiles = smiles
        self.labels = labels

    def __len__(self):
        return len(self.smiles)

class ECFPDataset(BaseDataset):
    """Dataset for single-task evaluation, returning ECFP fingerprints."""
    def __init__(self, smiles, labels, task_id=-1):
        super().__init__(smiles, labels)
        self.task_id = task_id
        self.featurizer = ECFPFeaturizer()

    def __getitem__(self, idx):
        s = self.smiles[idx]
        fp = self.featurizer(s)
        return {
            "vec": torch.tensor(fp),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }
    
    def get_task_id(self):
        return self.task_id

class SmilesDataset(BaseDataset):
    """Dataset for multi-task training, returning ECFP and weights."""
    def __init__(self, smiles, labels, weights):
        super().__init__(smiles, labels)
        self.weights = weights
        self.featurizer = ECFPFeaturizer()

    def __getitem__(self, idx):
        s = self.smiles[idx]
        fp = self.featurizer(s)
        batch_data = {
            "vec": torch.tensor(fp),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }
        weight = torch.tensor(self.weights[idx], dtype=torch.float32)
        return batch_data, weight

class OODDataset(BaseDataset):
    """Dataset for Out-of-Distribution data, returning only features."""
    def __init__(self, smiles):
        super().__init__(smiles)
        self.featurizer = ECFPFeaturizer()
    
    def __getitem__(self, idx):
        s = self.smiles[idx]
        fp = self.featurizer(s)
        return {"vec": torch.tensor(fp)}

# The following are variations that also return SMILES strings.
# They can be useful for debugging or analysis.
class OODDatasetSMI(OODDataset):
    """OOD Dataset that also returns the SMILES string."""
    def __getitem__(self, idx):
        s = self.smiles[idx]
        item = super().__getitem__(idx)
        item["smi"] = s
        return item

class ECFPDatasetSMI(ECFPDataset):
    """Single-task Dataset that also returns the SMILES string."""
    def __getitem__(self, idx):
        s = self.smiles[idx]
        item = super().__getitem__(idx)
        item["smi"] = s
        return item

# --- Graph-based Datasets ---
# These datasets require torch_geometric

class GraphDataset(BaseDataset):
    """Dataset for single-task evaluation, returning graph objects."""
    def __init__(self, smiles, labels, task_id=-1):
        super().__init__(smiles, labels)
        self.task_id = task_id
        self.featurizer = GraphFeaturizer()

    def __getitem__(self, idx):
        s = self.smiles[idx]
        data = self.featurizer(s)
        data.y = torch.tensor(self.labels[idx], dtype=torch.long)
        return data

    def get_task_id(self):
        return self.task_id
        
class TrainGraphDataset(BaseDataset):
    """Dataset for multi-task training, returning graph objects with weights."""
    def __init__(self, smiles, labels, weights):
        super().__init__(smiles, labels)
        self.weights = weights
        self.featurizer = GraphFeaturizer()
        
    def __getitem__(self, idx):
        s = self.smiles[idx]
        data = self.featurizer(s)
        data.y = torch.tensor(self.labels[idx], dtype=torch.long)
        data.weight = torch.tensor(self.weights[idx], dtype=torch.float32)
        return data