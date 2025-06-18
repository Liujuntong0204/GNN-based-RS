import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import LGConv

class PDA_GNN(nn.Module):
    def __init__(self, num_nodes, embedding_dim=64, num_layers=2, alpha=[1.0,1.0,1.0]):
        super(PDA_GNN, self).__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        nn.init.kaiming_uniform_(self.embedding.weight, nonlinearity='linear')

        self.convs_title = nn.ModuleList([LGConv() for _ in range(num_layers)])
        self.convs_year = nn.ModuleList([LGConv() for _ in range(num_layers)])
        self.convs_cat = nn.ModuleList([LGConv() for _ in range(num_layers)])

        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)

    def forward(self, edge_index_title, edge_index_year, edge_index_cat, x=None):
        if x is None:
            x = self.embedding.weight
        x = F.normalize(x, dim=1)
        x_title = x
        for conv in self.convs_title:
            x_title = conv(x_title, edge_index_title)
        x_title = F.normalize(x_title, dim=1)
        
        x_year = x
        for conv in self.convs_year:
            x_year = conv(x_year, edge_index_year)
        x_year = F.normalize(x_year, dim=1)

        x_cat = x
        for conv in self.convs_cat:
            x_cat = conv(x_cat, edge_index_cat)
        x_cat = F.normalize(x_cat, dim=1)

        weights = F.softmax(self.alpha, dim=0)
        weights = torch.clamp(weights, min=1e-4)
        weights = weights / weights.sum()
        out = weights[0]*x_title + weights[1]*x_year + weights[2]*x_cat
        return out

    def get_embeddings(self):
        return self.embedding.weight.detach()
    
    def get_final_embeddings(self, edge_index_title, edge_index_year, edge_index_cat):
        self.eval()
        with torch.no_grad():
            return self.forward(edge_index_title, edge_index_year, edge_index_cat)
        

def bpr_loss(emb, pos_indices, neg_indices):
    i, j = pos_indices
    k_i, k = neg_indices
    pos_score = (emb[i] * emb[j]).sum(1)
    neg_score = (emb[k_i] * emb[k]).sum(1)
    diff = pos_score - neg_score

    return -F.logsigmoid(diff).mean()