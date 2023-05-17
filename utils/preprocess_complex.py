# File Name: graph_constructor.py
# E-mail: jiang_dj@zju.edu.cn
from rdkit.Chem import rdmolfiles, rdmolops
from rdkit import Chem
import dgl
from scipy.spatial import distance_matrix
import numpy as np
import torch

from dgllife.utils import BaseAtomFeaturizer, atom_type_one_hot, atom_degree_one_hot, atom_total_num_H_one_hot, \
    atom_is_aromatic, ConcatFeaturizer, bond_type_one_hot, atom_hybridization_one_hot, \
    one_hot_encoding, atom_formal_charge, atom_num_radical_electrons, bond_is_conjugated, \
    bond_is_in_ring, bond_stereo_one_hot, BaseBondFeaturizer

import pickle
import os
#from dgl.data.chem import BaseBondFeaturizer
from functools import partial
import warnings
import dgl.backend as F
from dgl.data.utils import save_graphs, load_graphs
import multiprocessing
from itertools import repeat
warnings.filterwarnings('ignore')
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import torch
import pickle
from dgl.data import DGLDataset
from biopandas.pdb import PandasPdb
import dgl
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def get_distance_matrix(coords):
    diff_tensor = np.expand_dims(coords, axis=1) - np.expand_dims(coords, axis=0)
    distance_matrix = np.sqrt(np.sum(np.power(diff_tensor, 2), axis=-1))
    return distance_matrix
def transfer_residue_name(residue_name):
    data = ['ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'MET', 'TRP', 'PHE', 'GLN',
            'SER', 'THR', 'CYS', 'ASN', 'TYR', 'GLY', 'ASP', 'GLU', 'LYS',
            'ARG', 'HIS']
    # ohe = OneHotEncoder()  # 初始化独热编码
    # ohe.fit_transform(data).toarray()  # 拟合
    if residue_name.isdigit():
        return residue_name
    return data.index(residue_name)

def chirality(atom):  # the chirality information defined in the AttentiveFP
    try:
        return one_hot_encoding(atom.GetProp('_CIPCode'), ['R', 'S']) + \
               [atom.HasProp('_ChiralityPossible')]
    except:
        return [False, False] + [atom.HasProp('_ChiralityPossible')]

class MyAtomFeaturizer_0(BaseAtomFeaturizer):
    #用于产生g，只考虑骨架
    def __init__(self, atom_data_filed='h'):
        super(MyAtomFeaturizer_0, self).__init__(
            featurizer_funcs={atom_data_filed: ConcatFeaturizer([partial(atom_type_one_hot,
                                                                         allowable_set=['C', 'N', 'O'],
                                                                         encode_unknown=True)
                                                                 ])})
class MyAtomFeaturizer(BaseAtomFeaturizer):
    def __init__(self, atom_data_filed='h'):
        super(MyAtomFeaturizer, self).__init__(
            featurizer_funcs={atom_data_filed: ConcatFeaturizer([partial(atom_type_one_hot,
                                                                         allowable_set=['C', 'N', 'O', 'S', 'F', 'P',
                                                                                        'Cl', 'Br', 'I', 'B', 'Si',
                                                                                        'Fe', 'Zn', 'Cu', 'Mn', 'Mo'],
                                                                         encode_unknown=True),
                                                                 partial(atom_degree_one_hot,
                                                                         allowable_set=list(range(6))),
                                                                 atom_formal_charge, atom_num_radical_electrons,
                                                                 partial(atom_hybridization_one_hot,
                                                                         encode_unknown=True),
                                                                 atom_is_aromatic,
                                                                 # A placeholder for aromatic information,
                                                                 atom_total_num_H_one_hot, chirality])})

class MyBondFeaturizer_0(BaseBondFeaturizer):
    def __init__(self, bond_data_filed='e'):
        super(MyBondFeaturizer_0, self).__init__(
            featurizer_funcs={bond_data_filed: ConcatFeaturizer([bond_type_one_hot])})

class MyBondFeaturizer(BaseBondFeaturizer):
    def __init__(self, bond_data_filed='e'):
        super(MyBondFeaturizer, self).__init__(
            featurizer_funcs={bond_data_filed: ConcatFeaturizer([bond_type_one_hot, bond_is_conjugated, bond_is_in_ring,
                                                                 partial(bond_stereo_one_hot, allowable_set=[
                                                                     Chem.rdchem.BondStereo.STEREONONE,
                                                                     Chem.rdchem.BondStereo.STEREOANY,
                                                                     Chem.rdchem.BondStereo.STEREOZ,
                                                                     Chem.rdchem.BondStereo.STEREOE],
                                                                         encode_unknown=True)])})


def D3_info(a, b, c):
    # 空间夹角
    ab = b - a  # 向量ab
    ac = c - a  # 向量ac
    cosine_angle = np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac))
    cosine_angle = cosine_angle if cosine_angle >= -1.0 else -1.0
    angle = np.arccos(cosine_angle)
    # 三角形面积
    ab_ = np.sqrt(np.sum(ab ** 2))
    ac_ = np.sqrt(np.sum(ac ** 2))  # 欧式距离
    area = 0.5 * ab_ * ac_ * np.sin(angle)
    return np.degrees(angle), area, ac_


# claculate the 3D info for each directed edge
def D3_info_cal(nodes_ls, g):
    if len(nodes_ls) > 2:
        Angles = []
        Areas = []
        Distances = []
        for node_id in nodes_ls[2:]:
            angle, area, distance = D3_info(g.ndata['pos'][nodes_ls[0]].numpy(), g.ndata['pos'][nodes_ls[1]].numpy(),
                                            g.ndata['pos'][node_id].numpy())
            Angles.append(angle)
            Areas.append(area)
            Distances.append(distance)
        return [np.max(Angles) * 0.01, np.sum(Angles) * 0.01, np.mean(Angles) * 0.01, np.max(Areas), np.sum(Areas),
                np.mean(Areas),
                np.max(Distances) * 0.1, np.sum(Distances) * 0.1, np.mean(Distances) * 0.1]
    else:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]


AtomFeaturizer = MyAtomFeaturizer()
AtomFeaturizer_0 = MyAtomFeaturizer_0()
BondFeaturizer = MyBondFeaturizer()
BondFeaturizer_0 = BaseBondFeaturizer({'bond_type': bond_type_one_hot, 'in_ring': bond_is_in_ring})


def graphs_from_mol_mul(m1, m2, dis_threshold=8.0):
    """
    This function is used for generating graph objects using multi-process
    :param dir: the absoute path for the complex
    :param key: the key for the complex
    :param label: the label for the complex
    :param dis_threshold: the distance threshold to determine the atom-pair interactions
    :param graph_dic_path: the absoute path for storing the generated graph
    :param path_marker: '\\' for window and '/' for linux
    :return:
    """

    add_self_loop = False
    try:

        mol1, mol2 = m1, m2
        # the distance threshold to determine the interaction between ligand atoms and protein atoms
        dis_threshold = dis_threshold
        # small molecule
        # mol1 = m1
        # pocket
        # mol2 = m2

        # construct graphs1
        g = dgl.DGLGraph()
        # add nodes
        num_atoms_m1 = mol1.GetNumAtoms()  # number of ligand atoms
        num_atoms_m2 = mol2.GetNumAtoms()  # number of pocket atoms
        num_atoms = num_atoms_m1 + num_atoms_m2
        g.add_nodes(num_atoms)

        if add_self_loop:
            nodes = g.nodes()
            g.add_edges(nodes, nodes)

        # add edges, ligand molecule
        num_bonds1 = mol1.GetNumBonds()
        src1 = []
        dst1 = []
        for i in range(num_bonds1):
            bond1 = mol1.GetBondWithIdx(i)
            u = bond1.GetBeginAtomIdx()
            v = bond1.GetEndAtomIdx()
            src1.append(u)
            dst1.append(v)
        src_ls1 = np.concatenate([src1, dst1])
        dst_ls1 = np.concatenate([dst1, src1])
        g.add_edges(src_ls1, dst_ls1)

        # add edges, pocket
        num_bonds2 = mol2.GetNumBonds()
        src2 = []
        dst2 = []
        for i in range(num_bonds2):
            bond2 = mol2.GetBondWithIdx(i)
            u = bond2.GetBeginAtomIdx()
            v = bond2.GetEndAtomIdx()
            src2.append(u + num_atoms_m1)
            dst2.append(v + num_atoms_m1)
        src_ls2 = np.concatenate([src2, dst2])
        dst_ls2 = np.concatenate([dst2, src2])
        g.add_edges(src_ls2, dst_ls2)

        # add interaction edges, only consider the euclidean distance within dis_threshold
        g3 = dgl.DGLGraph()
        g3.add_nodes(num_atoms)
        dis_matrix = distance_matrix(mol1.GetConformers()[0].GetPositions(), mol2.GetConformers()[0].GetPositions())
        node_idx = np.where(dis_matrix < dis_threshold)
        src_ls3 = np.concatenate([node_idx[0], node_idx[1] + num_atoms_m1])
        dst_ls3 = np.concatenate([node_idx[1] + num_atoms_m1, node_idx[0]])
        g3.add_edges(src_ls3, dst_ls3)

        # assign atom features
        # 'h', features of atoms
        g.ndata['h'] = torch.zeros(num_atoms, AtomFeaturizer.feat_size('h'))  # init 'h'
        g.ndata['h'][:num_atoms_m1] = AtomFeaturizer(mol1)['h']
        g.ndata['h'][-num_atoms_m2:] = AtomFeaturizer(mol2)['h']

        # assign edge features
        # 'd', distance between ligand atoms
        dis_matrix_L = distance_matrix(mol1.GetConformers()[0].GetPositions(), mol1.GetConformers()[0].GetPositions())
        m1_d = torch.tensor(dis_matrix_L[src_ls1, dst_ls1], dtype=torch.float).view(-1, 1)

        # 'd', distance between pocket atoms
        dis_matrix_P = distance_matrix(mol2.GetConformers()[0].GetPositions(), mol2.GetConformers()[0].GetPositions())
        m2_d = torch.tensor(dis_matrix_P[src_ls2 - num_atoms, dst_ls2 - num_atoms_m1], dtype=torch.float).view(-1, 1)

        # 'd', distance between ligand atoms and pocket atoms
        inter_dis = np.concatenate([dis_matrix[node_idx[0], node_idx[1]], dis_matrix[node_idx[0], node_idx[1]]])
        g3_d = torch.tensor(inter_dis, dtype=torch.float).view(-1, 1)

        # efeats1
        g.edata['e'] = torch.zeros(g.number_of_edges(), BondFeaturizer.feat_size('e'))  # init 'h'
        efeats1 = BondFeaturizer(mol1)['e']  # 重复的边存在！
        g.edata['e'][g.edge_ids(src_ls1, dst_ls1)] = torch.cat([efeats1[::2], efeats1[::2]])

        # efeats2
        efeats2 = BondFeaturizer(mol2)['e']  # 重复的边存在！
        g.edata['e'][g.edge_ids(src_ls2, dst_ls2)] = torch.cat([efeats2[::2], efeats2[::2]])

        # 'e'
        g1_d = torch.cat([m1_d, m2_d])
        g.edata['e'] = torch.cat([g.edata['e'], g1_d * 0.1], dim=-1)
        g3.edata['e'] = g3_d * 0.1

        # if add_3D:
        # init 'pos'
        g.ndata['pos'] = torch.zeros([g.number_of_nodes(), 3])
        g.ndata['pos'][:num_atoms_m1] = torch.tensor(mol1.GetConformers()[0].GetPositions(), dtype=torch.float)
        g.ndata['pos'][-num_atoms_m2:] = torch.tensor(mol2.GetConformers()[0].GetPositions(), dtype=torch.float)
        # calculate the 3D info for g
        src_nodes, dst_nodes = g.find_edges(range(g.number_of_edges()))
        src_nodes, dst_nodes = src_nodes.tolist(), dst_nodes.tolist()
        neighbors_ls = []
        for i, src_node in enumerate(src_nodes):
            tmp = [src_node, dst_nodes[i]]  # the source node id and destination id of an edge
            neighbors = g.predecessors(src_node).tolist()
            neighbors.remove(dst_nodes[i])
            tmp.extend(neighbors)
            neighbors_ls.append(tmp)
        D3_info_ls = list(map(partial(D3_info_cal, g=g), neighbors_ls))
        D3_info_th = torch.tensor(D3_info_ls, dtype=torch.float)
        g.edata['e'] = torch.cat([g.edata['e'], D3_info_th], dim=-1)
        g.ndata.pop('pos')
        # detect the nan values in the D3_info_th
        if torch.any(torch.isnan(D3_info_th)):
            status = False
            #print(key)
        else:
            status = True
    except:
        g = None
        g3 = None
        status = False
    if status:
        # linux
        # with open(graph_dic_path+key, 'wb') as f:
        #     pickle.dump({'g': g, 'g3': g3, 'key': key, 'label': label}, f)
        # window
        return g, g3

def graphs_from_mol_mul_protein(m1, dis_threshold=8.0):
    """
    This function is used for generating graph objects using multi-process
    :param dir: the absoute path for the complex
    :param key: the key for the complex
    :param label: the label for the complex
    :param dis_threshold: the distance threshold to determine the atom-pair interactions
    :param graph_dic_path: the absoute path for storing the generated graph
    :param path_marker: '\\' for window and '/' for linux
    :return:
    """
    status = True
    add_self_loop = False
    try:

        mol1 = m1
        # the distance threshold to determine the interaction between ligand atoms and protein atoms
        dis_threshold = dis_threshold
        # small molecule
        # mol1 = m1
        # pocket
        # mol2 = m2

        # construct graphs1
        g = dgl.DGLGraph()
        # add nodes
        num_atoms_m1 = mol1.GetNumAtoms()  # number of ligand atoms
        g.add_nodes(num_atoms_m1)

        if add_self_loop:
            nodes = g.nodes()
            g.add_edges(nodes, nodes)

        # add edges, ligand molecule
        num_bonds1 = mol1.GetNumBonds()
        src1 = []
        dst1 = []
        for i in range(num_bonds1):
            bond1 = mol1.GetBondWithIdx(i)
            u = bond1.GetBeginAtomIdx()
            v = bond1.GetEndAtomIdx()
            src1.append(u)
            dst1.append(v)
        src_ls1 = np.concatenate([src1, dst1])
        dst_ls1 = np.concatenate([dst1, src1])
        g.add_edges(src_ls1, dst_ls1)



    except:
        g = None
        g3 = None
        status = False
    if status:
        # linux
        # with open(graph_dic_path+key, 'wb') as f:
        #     pickle.dump({'g': g, 'g3': g3, 'key': key, 'label': label}, f)
        # window
        return g


