import datetime
import numpy
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import numpy as np
from tqdm import tqdm
import argparse
import dgl
import dgl.data
import torch
from utils.model import LGN
import random
import pickle
from torch.utils.data import Dataset, DataLoader
from prefetch_generator import BackgroundGenerator


def get_args():
    parser = argparse.ArgumentParser(description='Args for graph predition')
    parser.add_argument('-project_name', default='HMG-pld', help='project name')
    parser.add_argument('-seed', type=int, default=42, help='seed')
    parser.add_argument('-data', default='PDBBind_v2016_refined_set', help='data folder name')
    parser.add_argument('-data_val', default='PDBBind_v2016_core_set', help='evaluation data folder name')
    parser.add_argument('-num_epochs', type=int, default=20, help='epochs')
    parser.add_argument('-num_epochs_repetition', type=int, default=1, help='epochs')
    parser.add_argument('-batch_size', type=int, default=128, help='batch size')
    parser.add_argument('-lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('-w_d', type=float, default=0.000001, help='weight decay')
    parser.add_argument('-l_num', type=int, default=4, help='layer num')
    parser.add_argument('-h_dim', type=int, default=32, help='hidden dim')
    parser.add_argument('-drop_n', type=float, default=0.1, help='drop net')
    parser.add_argument('-drop_c', type=float, default=0.1, help='drop output')
    parser.add_argument('-device', type=int, default=0, help='device')
    parser.add_argument('-learn_eps', action="store_true", help='learn the epsilon weighting')
    parser.add_argument('-num_workers', type=int, default=0, help='number of workers')
    parser.add_argument('--min_lr', type=float, default=0.00001)
    args, _ = parser.parse_known_args()
    return args



def sep_data(labels):
    test_idx = []
    train_idx = []
    labels = labels
    for l in range(len(labels)):
        if labels[l] < 0:
            test_idx.append(l)
        else:
            train_idx.append(l)
    train_idx = list(train_idx)
    return train_idx, test_idx



def train(args, model, loss_fcn, dataloader, optimizer, device):
    model.train()
    batch_pred_molecule = []
    batch_labels_molecule = []
    for code, ligand, bg, bg3, fp_s, fp_e, labels in dataloader:
        for i in range(len(ligand)):
            ligand[i] = ligand[i].to(device)
        bg = bg.to(device)
        bg3 = bg3.to(device)
        fp = torch.cat((fp_s, fp_e), dim=1)
        fp = fp.to(device)
        labels = torch.tensor(labels).to(device)
        batch_pred = model(ligand, bg, bg3, fp)
        batch_pred = batch_pred.squeeze()
        batch_labels = labels.squeeze()
        train_loss = loss_fcn(batch_pred.to(device), batch_labels.to(device))

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        for i in range(len(batch_labels)):
            l = batch_labels[i].cpu().numpy().astype(numpy.float32)
            if l != 0:
                batch_labels_molecule.append(l)
                batch_pred_molecule.append(batch_pred[i].cpu().detach().numpy().astype(numpy.float32))

    train_pred = np.array(batch_pred_molecule)
    train_label = np.array(batch_labels_molecule)


    return train_loss, train_label, train_pred

def evaluate(args, model, test_dataloader, device):
    model.eval()
    with torch.no_grad():
        batch_code = []
        batch_pred = []
        batch_labels = []
        for code, ligand, bg, bg3, fp_s, fp_e, labels in test_dataloader:
            for i in range(len(ligand)):
                ligand[i] = ligand[i].to(device)
            bg = bg.to(device)
            bg3 = bg3.to(device)
            fp = torch.cat((fp_s, fp_e), dim=1)
            fp = fp.to(device)
            labels = torch.tensor(labels).to(device)
            pred = model(ligand, bg, bg3, fp)
            pred = pred.squeeze()
            labels = labels.squeeze()
            for i in range(len(labels)):
                l = labels[i].cpu().numpy().astype(numpy.float32)
                c = code[i]

                if l != 0:
                    l = -l
                    batch_code.append(c)
                    batch_labels.append(l)
                    batch_pred.append(pred[i].cpu().numpy().astype(numpy.float32))
    return batch_code, batch_labels, batch_pred


class My_Dataset(Dataset):
    def __init__(self, graph):
        self.graph = graph
    def __len__(self):
        return len(self.graph)
    def __getitem__(self, index):

        return self.graph[index].code, self.graph[index].ligand, self.graph[index].g, self.graph[index].g3,self.graph[index].fp_s, self.graph[index].fp_e, torch.tensor(self.graph[index].label)
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def collate_fn(data_batch):
    code, ligands, graphs, graphs3, fp_s, fp_e, label = map(list, zip(*data_batch))
    bg = dgl.batch(graphs)
    bg3 = dgl.batch(graphs3)
    label = torch.tensor(label)
    fp_s = torch.stack(fp_s, dim = 0)
    fp_e = torch.stack(fp_e, dim=0)
    return code, ligands, bg, bg3, fp_s, fp_e, label

class MyLoss(torch.nn.Module):
    def __init__(self, alph):
        super(MyLoss, self).__init__()
        self.alph = alph

    def forward(self, input, target):
        sum_xy = torch.sum(torch.sum(input * target))
        sum_x = torch.sum(torch.sum(input))
        sum_y = torch.sum(torch.sum(target))
        sum_x2 = torch.sum(torch.sum(input * input))
        sum_y2 = torch.sum(torch.sum(target * target))
        n = input.size()[0]
        pcc = (n * sum_xy - sum_x * sum_y) / torch.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
        return self.alph*(1-torch.abs(pcc)) + (1-self.alph)*torch.nn.functional.mse_loss(input, target)

def main():
    args = get_args()
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(nowtime)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    input_file_path = 'dataset/preprocessed_' + args.data + '_' + args.data_val
    with open(input_file_path, 'rb') as input_file:
        print(input_file)
        graphs = pickle.load(input_file)
    print('graphs： ', len(graphs))
    labels = []
    for graph in graphs:
         graph.ligand.ndata['feat'] = graph.ligand.ndata['feat'].float()
         labels.append(graph.label)

    train_idx, test_idx = sep_data(labels)
    node_feat_size = 40
    edge_feat_size = 21
    in_feats = graphs[0].ligand.ndata['feat'].size()[1]
    gnn_output_dim = 16
    ign_output_dim = 64

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False

    model = LGN(args.l_num, 2, in_feats, args.h_dim, gnn_output_dim, ign_output_dim, 1, args.drop_c,
                        args.learn_eps, 'sum', 'sum', node_feat_size, edge_feat_size).to(device)

    train_index = train_idx
    train_data = []
    for train_i in range(len(graphs)):
        if train_i in train_index:
            train_data.append(graphs[train_i])

    train_dataset = My_Dataset(train_data)
    train_dataloader = DataLoaderX(train_dataset, batch_size = args.batch_size, shuffle=False, num_workers=args.num_workers,
                               collate_fn=collate_fn)
    print('train data: ',len(train_data))
    test_data = []
    for test_i in range(len(graphs)):
        if test_i in test_idx:
            test_data.append(graphs[test_i])
    test_dataset = My_Dataset(test_data)
    test_dataloader = DataLoaderX(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                   collate_fn=collate_fn)
    print('test data: ', len(test_data))

    loss_fcn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_d)
    for epoch in tqdm(range(args.num_epochs), desc='epochs', unit='epoch'):
        train_loss, train_label, train_pred = train(args, model,loss_fcn,train_dataloader, optimizer, device)
        train_r, train_p_value = pearsonr(train_label, train_pred)
        train_rmse = np.sqrt(mean_squared_error(train_label, train_pred))
        train_mae = mean_absolute_error(train_label, train_pred)

    print("trainning set")
    print('Rp: ', train_r)
    print('RMSE: ', train_rmse)
    print('MAE: ', train_mae)

    test_code, test_label, test_pred = evaluate(args, model,test_dataloader, device)
    test_r, test_p_value = pearsonr(test_label, test_pred)
    test_rmse = np.sqrt(mean_squared_error(test_label, test_pred))
    test_mae = mean_absolute_error(test_label, test_pred)
    print("test set")
    print('Rp: ', test_r)
    print('RMSE: ', test_rmse)
    print('MAE: ', test_mae)


if __name__ == '__main__':
    main()
