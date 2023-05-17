import pickle
import argparse
import torch
from utils.data_loader import FileLoader


ngpu= 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


def get_args():
    parser = argparse.ArgumentParser(description='Args for graph predition')
    parser.add_argument('-data', default='PDBBind_v2016_refined_set', help='data folder name')
    parser.add_argument('-data_val', default='PDBBind_v2016_core_set', help='data folder name')
    args, _ = parser.parse_known_args()
    return args





args = get_args()

data = FileLoader(args).load_data()
with open('dataset/preprocessed_' + args.data + '_' + args.data_val, 'wb') as save_file:
    pickle.dump(data, save_file)



