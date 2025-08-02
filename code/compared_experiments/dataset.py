"""
@author: Xie Xingran
@Date: 21-12-27
"""


from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import BatchSampler
import torch
import random
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import numpy as np
from torch_geometric.data import Data

families_dict = {
    'KDM': ['JMJD2', 'KDM4E', 'LSD1', 'KDM4C', 'KDM5B', 'KDM5A', 'KDM2B', 'KDM5C', 'KDM4B', 'KDM2A', 'PHF8', 'JMJD3', 'JMJD4', 'KDM4D', 'LSD2'],
    'HDAC': ['HDAC1', 'HDAC6', 'HDAC3', 'HDAC8', 'HDAC2', 'SIRT2', 'SIRT1', 'HDAC11', 'HDAC10', 'HDAC4', 'HDAC7', 'HDAC5', 'HDAC9', 'SIRT3', 'SIRT6', 'SIRT7'],
    'HAT': ['KAT2A', 'CREBBP', 'NCOA3', 'P300', 'NCOA1', 'PCAF', 'MYST1'],
    'PMT': ['EZH2', 'PRMT5', 'NSD2', 'PRMT4', 'DOT1L', 'PRMT6', 'SMYD3', 'PRMT1', 'PRMT3', 'SMYD2', 'PRMT8', 'SETD8', 'SETD7', 'SUV39H1', 'G9a'],
    'DNMT': ['DNMT3A', 'DNMT1', 'DNMT3B'],
    'reader': ['BAZ1A', 'BAZ2A', 'BRD8', 'PBRM1', 'BRWD1', 'SMARCA4', 'BRD9', 'BRPF1', 'TP53BP1', 'CBX7', 'BRD2', 'BRD7', 'L3MBTL3', 'L3MBTL1', 'BRD4', 'BRDT', 'BRPF3', 'WRD5', 'BAZ2B', 'BRD1', 'SMARCA2', 'BRD3']
}

idx2name_dict = {
    0: 'JMJD2',
    1: 'KAT2A',
    2: 'KDM4E',
    3: 'HDAC1',
    4: 'HDAC6',
    5: 'HDAC3',
    6: 'HDAC8',
    7: 'HDAC2',
    8: 'LSD1',
    9: 'SIRT2',
    10: 'KDM4C',
    11: 'EZH2',
    12: 'KDM5B',
    13: 'KDM5A',
    14: 'PRMT5',
    15: 'SIRT1',
    16: 'HDAC11',
    17: 'KDM2B',
    18: 'HDAC10',
    19: 'CREBBP',
    20: 'NSD2',
    21: 'HDAC4',
    22: 'HDAC7',
    23: 'HDAC5',
    24: 'NCOA3',
    25: 'HDAC9',
    26: 'PRMT4',
    27: 'SIRT3',
    28: 'P300',
    29: 'KDM5C',
    30: 'DOT1L',
    31: 'PRMT6',
    32: 'KDM4B',
    33: 'SMYD3',
    34: 'KDM2A',
    35: 'PRMT1',
    36: 'PHF8',
    37: 'PRMT3',
    38: 'JMJD3',
    39: 'NCOA1',
    40: 'PCAF',
    41: 'SMYD2',
    42: 'PRMT8',
    43: 'JMJD4',
    44: 'KDM4D',
    45: 'SETD8',
    46: 'SIRT6',
    47: 'SETD7',
    48: 'LSD2',
    49: 'SIRT7',
    50: 'SUV39H1',
    51: 'MYST1',
    52: 'BAZ1A',
    53: 'BAZ2A',
    54: 'BRD8',
    55: 'PBRM1',
    56: 'BRWD1',
    57: 'SMARCA4',
    58: 'DNMT3A',
    59: 'BRD9',
    60: 'BRPF1',
    61: 'TP53BP1',
    62: 'CBX7',
    63: 'BRD2',
    64: 'BRD7',
    65: 'L3MBTL3',
    66: 'L3MBTL1',
    67: 'BRD4',
    68: 'BRDT',
    69: 'BRPF3',
    70: 'DNMT1',
    71: 'WRD5',
    72: 'BAZ2B',
    73: 'BRD1',
    74: 'SMARCA2',
    75: 'BRD3',
    76: 'DNMT3B',
    77: 'G9a'
}

allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)),
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list' : [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}

def mol_to_graph_data_obj_simple(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    num_atom_features = 2   # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
            'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2   # bond type, bond direction
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                                            'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data

class OODDataset(Dataset):
    def __init__(self, smiles):
        super().__init__()
        
        self.smiles = smiles
    
    def getECFP(self, smiles):
        '''
        compute ECFP4 by rdkit
        @return: 
            ECFP4 -> np.array
        '''
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        arr = np.zeros((0,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)

        return arr
    
    def process(self, s):
        return self.getECFP(s)
    
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        '''
        @return:
            {key: "vec", val: ECPF4 -> torch.tensor(b x 1024)
            key: "labels", val: active(1) or inactive(0) -> tensor(b x n)} -> dict
            weights -> tensor(b x n)
        '''
        # print(item)
        s = self.smiles[idx]
        fp = self.process(s)
        
        return {"vec": torch.tensor(fp) if not torch.is_tensor(fp) else fp}

class SmilesDataset(Dataset):
    '''
    each compound has n labels,
    n mean the total number of task,
    b means the total number of compounds.
    '''
    def __init__(self, smiles, labels, weights):
        '''
        @Params:
            simels -> strings (1 x b)
            labels -> np.array (b x n)
            weight -> np.array (b x n) : 
                the index of corespoding task with actual datapoint(has label 1 or 0) will be set to 1, 
                else(has no label) will be set to 0
        '''
        self.smiles = smiles
        self.labels = labels
        self.weights = weights
    
    def __len__(self):
        '''
        get total number of compounds
        '''
        return len(self.smiles)

    def __getitem__(self, idx):
        '''
        @return:
            {key: "vec", val: ECPF4 -> torch.tensor(b x 1024)
            key: "labels", val: active(1) or inactive(0) -> tensor(b x n)} -> dict
            weights -> tensor(b x n)
        '''
        # print(item)
        s = self.smiles[idx]
        fp = self.process(s)
        output = {"vec": torch.tensor(fp) if not torch.is_tensor(fp) else fp,
                  "labels": self.labels[idx]}
        weight = self.weights[idx]         

        return {key: value for key, value in output.items()}, weight
    
    def process(self, s):
        
        return self.getECFP(s)
        
    def getECFP(self, smiles):
        '''
        compute ECFP4 by rdkit
        @return: 
            ECFP4 -> np.array
        '''
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        arr = np.zeros((0,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)

        return arr

class MPNNSmilesDataset(Dataset):
    '''
    each compound has n labels,
    n mean the total number of task,
    b means the total number of compounds.
    '''
    def __init__(self, smiles, labels, weights):
        '''
        @Params:
            simels -> strings (1 x b)
            labels -> np.array (b x n)
            weight -> np.array (b x n) : 
                the index of corespoding task with actual datapoint(has label 1 or 0) will be set to 1, 
                else(has no label) will be set to 0
        '''
        self.smiles = smiles
        self.labels = labels
        self.weights = weights
    
    def __len__(self):
        '''
        get total number of compounds
        '''
        return len(self.smiles)

    def __getitem__(self, idx):
        '''
        @return:
            {key: "vec", val: ECPF4 -> torch.tensor(b x 1024)
            key: "labels", val: active(1) or inactive(0) -> tensor(b x n)} -> dict
            weights -> tensor(b x n)
        '''
        # print(item)
        s = self.smiles[idx]
        output = {"smiles": s,
                  "labels": self.labels[idx]}
        weight = self.weights[idx]         

        return s, self.labels[idx], weight

class ECFPDataset(Dataset):
    def __init__(self, smiles, labels, task_id=-1):
        self.smiles = smiles
        self.labels = labels 
        # use to distinguish different task
        self.task_id = task_id

    def __len__(self):
        return len(self.smiles)
        
    def __getitem__(self, idx):
        '''
        @return:
            {key: "vec", val: ECPF4 -> torch.tensor
            key: "label", val: active(1) or inactive(0) -> tensor}
        '''
        # print(item)
        
        s = self.smiles[idx]
        fp = self.process(s)
        output = {"vec": torch.tensor(fp) if not torch.is_tensor(fp) else fp,
                  "label": self.labels[idx]}         

        return {key: value for key, value in output.items()}

    # def load(self, index):
    #     self.ecfp = [self.getECFP(self.smiles[i]) for i in index]
    #     self.labels = [self.labels[i] for i in index]
    #     # self.corpus_lines = len(self.smiles)
    #     return self

    def process(self, s):
        
        return self.getECFP(s)
    
    def get_task_id(self):
        return self.task_id
        
    def getECFP(self, smiles):
        '''
        compute ECFP4 by rdkit
        @return: 
        ECFP4 -> np.array
        '''
        if Chem.MolFromSmiles(smiles) is None:
            return np.array([0] * 1024, dtype=np.float32)
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        arr = np.zeros((0,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)

        return arr

class GraphDataset(Dataset):
    def __init__(self, smiles, labels, weights=None, task_id=None):
        self.smiles = smiles
        self.labels = labels
        self.weights = weights
        self.task_id = task_id if task_id is not None else -1  # default value for task_id
        # use to distinguish different task

    def __len__(self):
        return len(self.smiles)
        
    def __getitem__(self, idx):
        '''
        @return:
            data.x: node feature
            data.edge_index: edge index
            data.edge_atrr: edge feature
            data.y : compound labels, val: active(1) or inactive(0) -> tensor(b x n)
        '''
        
        s = self.smiles[idx]

        data = self.getGraphData(s)

        data.y = torch.tensor(self.labels[idx])
        if self.weights is not None:
            # If weights are provided, add them to the data object
            if isinstance(self.weights[idx], torch.Tensor):
                data.weights = self.weights[idx]
            else:
                # Convert to tensor if weights are not already a tensor
                data.weights = torch.tensor(self.weights[idx], dtype=torch.float32)
        return data

    def get_task_id(self):
        return self.task_id
    
    # def load(self, index):
    #     self.ecfp = [self.getECFP(self.smiles[i]) for i in index]
    #     self.labels = [self.labels[i] for i in index]
    #     # self.corpus_lines = len(self.smiles)
    #     return self
        
    def getGraphData(self, smiles):
        '''
        process smiles to graph data
        @return:
            data:include node featrue, edge featrue, edge index,
        '''
        mol = Chem.MolFromSmiles(smiles)
        data = mol_to_graph_data_obj_simple(mol)

        return data

# class GraphDataset(Dataset):
#     '''
#     each compound has n labels,
#     n mean the total number of task,
#     b means the total number of compounds.
#     '''
#     def __init__(self, smiles, labels, task_id):
#         '''
#         @Params:
#             simels -> strings (1 x b)
#             labels -> np.array (b x n)
#             weight -> np.array (b x n) : 
#                 the index of corespoding task with actual datapoint(has label 1 or 0) will be set to 1, 
#                 else(has no label) will be set to 0
#         '''
#         self.smiles = smiles
#         self.labels = labels
#         self.task_id = task_id
    
#     def __len__(self):
#         '''
#         get total number of compounds
#         '''
#         return len(self.smiles)
    
#     def get_task_id(self):
#         return self.task_id
    
#     def __getitem__(self, idx):
#         '''
#         @return:
#             {key: "vec", val: ECPF4 -> torch.tensor
#             key: "label", val: active(1) or inactive(0) -> tensor}
#         '''
#         # print(item)
        
#         s = self.smiles[idx]
#         node_features, edge_features, adj_matrix = self.process(s)
#         # output = {"node_features": node_features,
#         #           "edge_features": edge_features,
#         #           "adj_matrix": adj_matrix,
#         #           "label": self.labels[idx]
#         #           }
        
#         data = Data(x=torch.tensor(np.array(node_features).squeeze(), dtype=torch.float32),
#                    edge_index=torch.tensor(np.array(adj_matrix.nonzero()), dtype=torch.long),
#                    edge_attr=torch.tensor(edge_features, dtype=torch.float32),
#                    y=self.labels[idx].clone().detach() if torch.is_tensor(self.labels[idx]) else torch.tensor(self.labels[idx], dtype=torch.float32)
#                    )

#         return data

#     # def load(self, index):
#     #     self.ecfp = [self.getECFP(self.smiles[i]) for i in index]
#     #     self.labels = [self.labels[i] for i in index]
#     #     # self.corpus_lines = len(self.smiles)
#     #     return self

#     def process(self, s):
#         node_features, edge_features, adj_matrix = self.getGraph(s)
#         return node_features, edge_features, adj_matrix
    
#     def get_atom_features(self, smiles):
#         """
#         从 SMILES 分子字符串中提取节点特征。
#         示例特征包括原子序数、形式电荷、是否芳香等。
#         """
#         mol = Chem.MolFromSmiles(smiles)
#         node_features = []
#         for atom in mol.GetAtoms():
#             features = [
#                 atom.GetAtomicNum(),                 # 原子序数
#                 atom.GetFormalCharge(),              # 形式电荷
#                 int(atom.GetIsAromatic()),           # 是否芳香
#                 atom.GetMass(),                      # 原子质量
#                 int(atom.GetHybridization()),        # 杂化类型 (0=UNSPECIFIED,1=S,2=SP,3=SP2…)
#                 atom.GetTotalNumHs(),                # 氢原子数
#             ]
            
#             # 根据需要添加更多信息
#             node_features.append([features])
#         return node_features
        
#     def getGraph(self, smiles):
#         '''
#         compute graph by rdkit
#         @return: 
#         graph -> np.array
#         '''
        
#         if Chem.MolFromSmiles(smiles) is None:
#             # num_atoms = mol.GetNumAtoms()
#             # num_bonds = mol.GetNumBonds()
#             node_features = np.zeros((2, 6), dtype=np.float32)
#             edge_features = np.zeros((4, 6), dtype=np.float32)
#             adj_matrix = np.eye(2, dtype=np.float32)
#             edge_features = np.concatenate([edge_features, edge_features], axis=0)
#             return node_features, edge_features, adj_matrix
        
#         mol = Chem.MolFromSmiles(smiles)
#         adj_matrix = Chem.GetAdjacencyMatrix(mol)
        
        
#         num_bonds = mol.GetNumBonds()
        
#         # node_features = np.zeros((num_atoms, 62), dtype=np.float32)
#         edge_features = np.zeros((num_bonds, 6), dtype=np.float32)

#         node_features = self.get_atom_features(smiles)

#         for bond in mol.GetBonds():
#             bond_idx = bond.GetIdx()
#             # 根据 bondTypeAsDouble 值填充对应位置
#             bond_type_idx = int(bond.GetBondTypeAsDouble())
#             edge_features[bond_idx, bond_type_idx] = 1
        
#         edge_features = np.concatenate([edge_features, edge_features], axis=0)
#         return node_features, edge_features, adj_matrix

class TrainGraphDataset(Dataset):
    '''
    each compound has n labels,
    n mean the total number of task,
    b means the total number of compounds.
    '''
    
    def __init__(self, smiles, labels, weights):
        '''
        @Params:
            simels -> strings (1 x b)
            labels -> np.array (b x n)
            weight -> np.array (b x n) : 
                the index of corespoding task with actual datapoint(has label 1 or 0) will be set to 1, 
                else(has no label) will be set to 0
        '''
        self.smiles = smiles
        self.labels = labels
        self.weights = weights
    
    def __len__(self):
        '''
        get total number of compounds
        '''
        return len(self.smiles)
    
    def __getitem__(self, idx):
        '''
        @return:
            {key: "vec", val: ECPF4 -> torch.tensor
            key: "label", val: active(1) or inactive(0) -> tensor}
        '''
        # print(item)
        
        s = self.smiles[idx]
        node_features, edge_features, adj_matrix = self.process(s)
        # output = {"node_features": node_features,
        #           "edge_features": edge_features,
        #           "adj_matrix": adj_matrix,
        #           "label": self.labels[idx]
        #           }
        weight = self.weights[idx]
        
        data = Data(x=torch.tensor(np.array(node_features).squeeze(), dtype=torch.float32),
                   edge_index=torch.tensor(np.array(adj_matrix.nonzero()), dtype=torch.long),
                   edge_attr=torch.tensor(edge_features, dtype=torch.float32),
                   y=torch.tensor(self.labels[idx], dtype=torch.float32),
                   )
        data.weight = torch.tensor(weight, dtype=torch.float32)

        return data

    # def load(self, index):
    #     self.ecfp = [self.getECFP(self.smiles[i]) for i in index]
    #     self.labels = [self.labels[i] for i in index]
    #     # self.corpus_lines = len(self.smiles)
    #     return self

    def process(self, s):
        node_features, edge_features, adj_matrix = self.getGraph(s)
        return node_features, edge_features, adj_matrix
    
    def get_atom_features(self, smiles):
        """
        从 SMILES 分子字符串中提取节点特征。
        示例特征包括原子序数、形式电荷、是否芳香等。
        """
        mol = Chem.MolFromSmiles(smiles)
        node_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),                 # 原子序数
                atom.GetFormalCharge(),              # 形式电荷
                int(atom.GetIsAromatic()),           # 是否芳香
                atom.GetMass(),                      # 原子质量
                int(atom.GetHybridization()),        # 杂化类型 (0=UNSPECIFIED,1=S,2=SP,3=SP2…)
                atom.GetTotalNumHs(),                # 氢原子数
            ]
            
            # 根据需要添加更多信息
            node_features.append([features])
        return node_features
    
    def getGraph(self, smiles):
        '''
        compute graph by rdkit
        @return: 
        graph -> np.array
        '''
        mol = Chem.MolFromSmiles(smiles)
        adj_matrix = Chem.GetAdjacencyMatrix(mol)
        
        # num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
        
        # node_features = np.zeros((num_atoms, 62), dtype=np.float32)
        edge_features = np.zeros((num_bonds, 6), dtype=np.float32)

        node_features = self.get_atom_features(smiles)

        for bond in mol.GetBonds():
            bond_idx = bond.GetIdx()
            # 根据 bondTypeAsDouble 值填充对应位置
            bond_type_idx = int(bond.GetBondTypeAsDouble())
            edge_features[bond_idx, bond_type_idx] = 1
        
        edge_features = np.concatenate([edge_features, edge_features], axis=0)
        return node_features, edge_features, adj_matrix


class MPNNTestSmilesDataset(Dataset):
    '''
    each compound has n labels,
    n mean the total number of task,
    b means the total number of compounds.
    '''
    def __init__(self, smiles, labels, task_id):
        '''
        @Params:
            simels -> strings (1 x b)
            labels -> np.array (b x n)
            weight -> np.array (b x n) : 
                the index of corespoding task with actual datapoint(has label 1 or 0) will be set to 1, 
                else(has no label) will be set to 0
        '''
        self.smiles = smiles
        self.labels = labels
        self.task_id = task_id
    
    def __len__(self):
        '''
        get total number of compounds
        '''
        return len(self.smiles)
    
    def get_task_id(self):
        return self.task_id

    def __getitem__(self, idx):
        '''
        @return:
            {key: "vec", val: ECPF4 -> torch.tensor(b x 1024)
            key: "labels", val: active(1) or inactive(0) -> tensor(b x n)} -> dict
            weights -> tensor(b x n)
        '''
        # print(item)
        s = self.smiles[idx]    

        return s, self.labels[idx]

class OODMPNNTestSmilesDataset(Dataset):
    '''
    each compound has n labels,
    n mean the total number of task,
    b means the total number of compounds.
    '''
    def __init__(self, smiles):
        '''
        @Params:
            simels -> strings (1 x b)
            labels -> np.array (b x n)
            weight -> np.array (b x n) : 
                the index of corespoding task with actual datapoint(has label 1 or 0) will be set to 1, 
                else(has no label) will be set to 0
        '''
        self.smiles = smiles
    
    def __len__(self):
        '''
        get total number of compounds
        '''
        return len(self.smiles)

    def __getitem__(self, idx):
        '''
        @return:
            {key: "vec", val: ECPF4 -> torch.tensor(b x 1024)
            key: "labels", val: active(1) or inactive(0) -> tensor(b x n)} -> dict
            weights -> tensor(b x n)
        '''
        # print(item)
        s = self.smiles[idx]    

        return s