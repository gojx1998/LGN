from utils.preprocess_complex import graphs_from_mol_mul, graphs_from_mol_mul_protein
import oddt
from oddt.fingerprints import SimpleInteractionFingerprint as SimpleInteractionFingerprint
import os
import deepchem as dc
from utils.preprocess_ligand import *
from utils.preprocess_fpe import *
import numpy as np


class Graph(object):
    def __init__(self, code, ligand, g, g3, fp_s, fp_e, label):
        self.code = code
        self.ligand = ligand
        self.g = g
        self.g3 = g3
        self.fp_s = fp_s
        self.fp_e = fp_e
        self.label = label

class FileLoader(object):
    def __init__(self, args):
        self.args = args


    def load_data(self):
        np.set_printoptions(threshold=1e6)
        data_path = self.args.data
        data_val_path = self.args.data_val
        Atom_Keys = pd.read_csv("utils/PDB_Atom_Keys.csv", sep=",")
        test_csv = pd.read_csv("utils/test_2020.csv", header=None)
        dataframe_val = pd.DataFrame(columns=['code', 'resolution', 'year', 'label'])
        # Load index file
        # 加载测试集数据index
        if data_val_path == 'PDBBind_v2016_core_set':
            path_val_index = "dataset\\CoreSet.dat"
            with open(path_val_index) as f:
                for _ in range(1): f.readline()
                index = f.readlines()[:-1]
            for data in index:
                data = data.split()[:5]
                code, resolution, year, label = data[0], data[1], int(data[2]), float(data[3])
                label = -label
                df = pd.DataFrame([[code, resolution, year, label]],
                                  columns=['code', 'resolution', 'year', 'label'])
                dataframe_val = dataframe_val.append(df, ignore_index=True, sort=True)

        print(len(dataframe_val))

        # 加载测试集数据
        if data_val_path == 'PDBBind_v2016_core_set':
            path_val = "dataset\\coreset"

        dataset = os.listdir(path_val)
        dataset.sort()
        Graph_list = []
        test_code = []
        test_label_list = []
        test_pocket_size_list = []
        test_ligand_size_list = []
        for idx, pdb_code in enumerate(dataset):
            # if idx == 100:
            #     break
            # 判断是否存在
            if pdb_code in dataframe_val['code'].values:
                graph_label = dataframe_val[(dataframe_val['code'] == pdb_code)]['label'].values
                l = graph_label[0].tolist()
            else:
                continue
            pdb_path = path_val + '/{}/{}'.format(pdb_code, pdb_code)
            ligand_path = pdb_path + '_ligand'
            ligand_file = ligand_path + '.mol2'
            pocket_path = pdb_path + '_pocket'
            pocket_file = pocket_path + '.pdb'
            protein_path = pdb_path + '_protein'
            protein_file = protein_path + '.pdb'

            if os.path.exists(ligand_file) and os.path.exists(pocket_file):
                try:
                    #获得g和g3
                    m1 = Chem.MolFromMol2File(ligand_file)
                    m2 = Chem.MolFromPDBFile(pocket_file)
                    g, g3 = graphs_from_mol_mul(m1, m2)
                    #获得ligand
                    c_mol = Chem.AddHs(m1, addCoords=True)
                    contacts_6A = GetAtomContacts(pocket_file, c_mol, Atom_Keys, distance_cutoff=6.0)
                    lig = mol_to_graph(c_mol, contacts_6A, Atom_Keys)
                    # 获得fingerprint
                    protein = next(oddt.toolkit.readfile('pdb', pocket_file))
                    protein.protein = True
                    ligand = next(oddt.toolkit.readfile('mol2', ligand_file))
                    fp_s = SimpleInteractionFingerprint(ligand, protein)
                    fp_s_num = fp_s
                    fp_s = torch.from_numpy(fp_s)
                    fp_e = GetECIF(pocket_file, ligand_file, distance_cutoff=6.0)  # 1540
                    fp_e = fp_e + GetELEMENTS(pocket_file, ligand_file, distance_cutoff=6.0)  # 1576
                    fp_e = fp_e + list(GetRDKitDescriptors(ligand_file))  # 1746
                    fp_e = torch.tensor(fp_e)
                    fp_s = fp_s.to(torch.float)
                    fp_e = fp_e.to(torch.float)

                    graph = Graph(pdb_code, lig, g, g3, fp_s, fp_e, l)
                    Graph_list.append(graph)

                except:
                    print('Parse error on {}. Skipping to next molecule'.format(pdb_code))
                    continue
            print('Converted {}: {}/{}'.format(pdb_code, idx, len(dataset)))

        test_df = pd.DataFrame({'code': test_code, 'true': test_label_list, 'pocket': test_pocket_size_list, 'ligand': test_ligand_size_list})
        test_df_path = './dataset/test_2.csv'
        test_df.to_csv(test_df_path, index=False)
        print('test set: ', len(Graph_list))

        # 加载训练集数据index
        dataframe = pd.DataFrame(columns=['code', 'resolution', 'year', 'label'])
        if data_path == 'PDBBind_v2016_refined_set':
            index_file_path = 'dataset\\INDEX_refined_data.2016'
            path = 'dataset\\PDBBind_v2016_refined_set'
        with open(index_file_path) as f:
                for _ in range(6): f.readline()
                index = f.readlines()[:-1]
        for data in index:
            data = data.split()[:5]
            code, resolution, year, label, type = data[0], data[1], int(data[2]), float(data[3]), data[4]
            if code in dataframe_val['code'].values:
                continue
            df = pd.DataFrame([[code, resolution, year, label]],
                              columns=['code', 'resolution', 'year', 'label'])
            dataframe = dataframe.append(df, ignore_index=True, sort=True)
        print('Index loaded')
        print(len(dataframe))

        dataset = os.listdir(path)
        dataset.sort()
        for idx, pdb_code in enumerate(dataset):
            if pdb_code in dataframe['code'].values:
                graph_label = dataframe[(dataframe['code'] == pdb_code)]['label'].values
                l = graph_label[0].tolist()
            else:
                continue
            pdb_path = path + '/{}/{}'.format(pdb_code, pdb_code)
            ligand_path = pdb_path + '_ligand'
            ligand_file = ligand_path + '.mol2'
            pocket_path = pdb_path + '_pocket'
            pocket_file = pocket_path + '.pdb'
            protein_path = pdb_path + '_protein'
            protein_file = protein_path + '.pdb'

            if os.path.exists(ligand_file) and os.path.exists(pocket_file):
                try:
                    m1 = Chem.MolFromMol2File(ligand_file)
                    m2 = Chem.MolFromPDBFile(pocket_file)
                    g, g3 = graphs_from_mol_mul(m1, m2)

                    c_mol = Chem.AddHs(m1, addCoords=True)
                    contacts_6A = GetAtomContacts(pocket_file, c_mol, Atom_Keys, distance_cutoff=6.0)
                    lig = mol_to_graph(c_mol, contacts_6A, Atom_Keys)

                    # 获得fingerprint
                    protein = next(oddt.toolkit.readfile('pdb', pocket_file))
                    protein.protein = True
                    ligand = next(oddt.toolkit.readfile('mol2', ligand_file))
                    fp_s = SimpleInteractionFingerprint(ligand, protein)
                    fp_s = torch.from_numpy(fp_s)
                    fp_e = GetECIF(pocket_file, ligand_file, distance_cutoff=6.0)  # 1540
                    fp_e = fp_e + GetELEMENTS(pocket_file, ligand_file, distance_cutoff=6.0)  # 1576
                    fp_e = fp_e + list(GetRDKitDescriptors(ligand_file))  # 1746
                    fp_e = torch.tensor(fp_e)
                    fp_s = fp_s.to(torch.float)
                    fp_e = fp_e.to(torch.float)

                    graph = Graph(pdb_code, lig, g, g3, fp_s, fp_e, l)
                    Graph_list.append(graph)

                except:
                    print('Parse error on {}. Skipping to next molecule'.format(pdb_code))
                    continue
            print('Converted {}: {}/{}'.format(pdb_code, idx, len(dataset)))

        print('test set and train set: ', len(Graph_list))
        return Graph_list

