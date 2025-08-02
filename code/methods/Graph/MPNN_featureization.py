# methods/Graph/MPNN_featureization.py

"""
Molecular graph featurization for Message Passing Neural Networks (MPNNs).

This module provides functions and classes to convert SMILES strings into
graph representations suitable for MPNN models. It defines how atoms and bonds
are featurized and how a batch of molecules is collated into a single large graph.
"""

import random
from argparse import Namespace
from copy import deepcopy
from typing import List, Tuple, Union

import torch
from rdkit import Chem

# Import constants from the dedicated module
from .MPNN_constants import ATOM_FEATURES, ATOM_FDIM, BOND_FDIM

# Memoization cache to speed up featurization of repeated SMILES
SMILES_TO_GRAPH = {}
SMILES_TO_CONTRA_GRAPH = {}

def clear_cache():
    """Clears the global featurization cache."""
    global SMILES_TO_GRAPH, SMILES_TO_CONTRA_GRAPH
    SMILES_TO_GRAPH = {}
    SMILES_TO_CONTRA_GRAPH = {}

def get_atom_fdim() -> int:
    """Returns the dimensionality of the atom feature vector."""
    return ATOM_FDIM

def get_bond_fdim() -> int:
    """Returns the dimensionality of the bond feature vector."""
    # Note: The actual feature vector used in the graph includes atom features.
    return BOND_FDIM

def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding for a given value.

    If the value is not in the list of choices, the last element of the encoding is set to 1.
    """
    encoding = [0] * (len(choices) + 1)
    try:
        index = choices.index(value)
    except ValueError:
        index = -1
    encoding[index] = 1
    return encoding

def atom_features(atom: Chem.rdchem.Atom) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a single atom.
    """
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
               onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
               onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
               onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
               onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
               onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
               [1 if atom.GetIsAromatic() else 0] + \
               [atom.GetMass() * 0.01]  # Scale mass to be in a similar range
    return features

def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a single bond.
    """
    if bond is None:
        # Special case for non-existent bonds
        return [1] + [0] * (BOND_FDIM - 1)
    
    bt = bond.GetBondType()
    features = [
        0,  # A flag indicating the bond is not None
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    features += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return features


class MolGraph:
    """
    Represents the graph structure and featurization of a single molecule.

    Attributes:
        smiles (str): The SMILES string of the molecule.
        n_atoms (int): The number of atoms.
        n_bonds (int): The number of directed bonds (edges), which is 2 * num_undirected_bonds.
        f_atoms (List[List]): A list of feature vectors for each atom.
        f_bonds (List[List]): A list of feature vectors for each directed bond.
        a2b (List[List[int]]): Maps an atom index to a list of incoming bond indices.
        b2a (List[int]): Maps a bond index to the index of its source atom.
        b2revb (List[int]): Maps a bond index to the index of its reverse bond.
    """

    def __init__(self, smiles: str):
        self.smiles = smiles
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES string: {smiles}")

        self.n_atoms = mol.GetNumAtoms()
        
        # Build atom features
        self._build_atom_features(mol)
        
        # Build bond features and graph structure
        self._build_bond_features_and_graph(mol)

    def _build_atom_features(self, mol: Chem.rdchem.Mol):
        """Computes feature vectors for each atom in the molecule."""
        self.f_atoms = [atom_features(atom) for atom in mol.GetAtoms()]

    def _build_bond_features_and_graph(self, mol: Chem.rdchem.Mol):
        """Builds the graph structure (bonds and connectivity mappings)."""
        self.n_bonds = 0
        self.f_bonds = []
        self.a2b = [[] for _ in range(self.n_atoms)]
        self.b2a = []
        self.b2revb = []

        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)
                if bond is None:
                    continue

                f_bond = bond_features(bond)
                
                # Add features for both directed edges
                self.f_bonds.append(self.f_atoms[a1] + f_bond)
                self.f_bonds.append(self.f_atoms[a2] + f_bond)

                # Update graph connectivity
                b1 = self.n_bonds
                b2 = b1 + 1
                
                # Edge from a1 to a2
                self.a2b[a2].append(b1)
                self.b2a.append(a1)
                
                # Edge from a2 to a1
                self.a2b[a1].append(b2)
                self.b2a.append(a2)
                
                # Map reverse bonds
                self.b2revb.append(b2)
                self.b2revb.append(b1)
                
                self.n_bonds += 2


class BatchMolGraph:
    """
    Collates a list of MolGraph objects into a single, batched graph representation.
    This allows for efficient processing of multiple molecules in parallel.
    """

    def __init__(self, mol_graphs: List[MolGraph]):
        self.smiles_batch = [mg.smiles for mg in mol_graphs]
        self.n_mols = len(mol_graphs)
        self.atom_fdim = get_atom_fdim()
        self.bond_fdim = get_bond_fdim() + self.atom_fdim # Combined feature vector

        # Initialize with padding for index 0
        self.n_atoms = 1
        self.n_bonds = 1
        self.a_scope = []
        self.b_scope = []

        # Zero-padded features and graph structure
        f_atoms = [[0] * self.atom_fdim]
        f_bonds = [[0] * self.bond_fdim]
        a2b = [[]]
        b2a = [0]
        b2revb = [0]

        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)

            # Update connectivity with offset for batching
            for a_idx in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a_idx]])
            for b_idx in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b_idx])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b_idx])

            # Record the scope (start index, number of items) for this molecule
            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            
            # Update total counts
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds

        # Pad a2b to be a rectangular tensor
        self.max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b))
        self.a2b = [bonds + [0] * (self.max_num_bonds - len(bonds)) for bonds in a2b]

        # Convert to PyTorch tensors
        self.f_atoms = torch.tensor(f_atoms, dtype=torch.float32)
        self.f_bonds = torch.tensor(f_bonds, dtype=torch.float32)
        self.a2b = torch.tensor(self.a2b, dtype=torch.long)
        self.b2a = torch.tensor(b2a, dtype=torch.long)
        self.b2revb = torch.tensor(b2revb, dtype=torch.long)

    def get_components(self) -> Tuple[torch.Tensor, ...]:
        """Returns the tensor components of the batched graph."""
        return self.f_atoms, self.f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope


def mol2graph(smiles_batch: List[str]) -> BatchMolGraph:
    """
    Converts a list of SMILES strings to a BatchMolGraph.
    Uses memoization to avoid re-computing graphs for the same SMILES.
    """
    mol_graphs = []
    for smiles in smiles_batch:
        if smiles not in SMILES_TO_GRAPH:
            SMILES_TO_GRAPH[smiles] = MolGraph(smiles)
        mol_graphs.append(SMILES_TO_GRAPH[smiles])
    
    return BatchMolGraph(mol_graphs)