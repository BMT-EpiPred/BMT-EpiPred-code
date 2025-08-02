# methods/Graph/MPNN_constants.py

"""
Constants for MPNN (Message Passing Neural Network) featurization.

This file defines the feature sets for atoms and bonds, and calculates
their resulting dimensionalities. Separating these constants improves the
clarity of the main featurization logic.
"""

from rdkit import Chem

# A list of atomic numbers that are explicitly considered in featurization.
# Others will be mapped to an "unknown" category.
MAX_ATOMIC_NUM = 100

# Dictionary defining the possible values for each atomic feature.
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
}

# Calculate the dimensionality of the atom feature vector.
# +1 for each feature set to accommodate "unknown" values.
# +2 at the end for 'IsAromatic' (boolean) and 'Mass' (float).
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2

# The number of features for a bond. This includes:
# - A flag indicating if the bond exists.
# - One-hot encoding for bond types (SINGLE, DOUBLE, TRIPLE, AROMATIC).
# - Flags for conjugation and ring membership.
# - One-hot encoding for stereo configuration.
BOND_FDIM = 14