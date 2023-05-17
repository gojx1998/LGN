import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model_ligand import GNN_L
from utils.model_complex import GNN_C


class FCL(nn.Module):
    def __init__(self, input_size, output_size):
        super(FCL, self).__init__()
        self.fcl = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, output):
        output = self.dropout(output)
        output = self.fcl(output)
        return output




class LGN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, pre_input_dim, hidden_dim,
                 gnn_output_dim, ign_output_dim, final_output_dim,dropout_0, learn_eps,
                 neighbor_pooling_type, graph_pooling_type, node_feat_size, edge_feat_size):
        super(LGN, self).__init__()
        self.device = torch.device('cuda:0')
        self.gnn = GNN_L(num_layers, num_mlp_layers, pre_input_dim, hidden_dim, gnn_output_dim, dropout_0, learn_eps, graph_pooling_type, neighbor_pooling_type, self.device).to(self.device)
        self.ign = GNN_C(node_feat_size, edge_feat_size, 2, 128, 128, 200, 2, 0.1, ign_output_dim)
        self.FC = nn.Linear(1914,128)
        self.prediction = FCL(gnn_output_dim+ign_output_dim+128, final_output_dim).to(self.device)

    def forward(self, ligand, bg, bg3, fp):
        pre_h = self.gnn(ligand)
        h = self.ign(bg, bg3)
        fp = self.FC(fp)
        h = torch.cat((h, pre_h, fp), 1)
        h = self.prediction(h)
        return h



