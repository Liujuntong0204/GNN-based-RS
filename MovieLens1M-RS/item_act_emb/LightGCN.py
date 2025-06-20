import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import LGConv

class LightGCN(nn.Module):
    def __init__(self, num_nodes, embedding_dim = 64, num_layers = 3, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.convs = nn.ModuleList([LGConv() for _ in range(num_layers)])
        self.layer_weights = nn.Parameter(torch.ones(num_layers + 1) / (num_layers + 1))
        self.dropout = dropout

    def forward(self, edge_index, edge_weight=None):
        x = F.dropout(self.embedding.weight, p=self.dropout, training=self.training)
        out = [x]
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            out.append(x)
        weights = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1)
        return F.normalize((weights * torch.stack(out)).sum(dim=0), dim=1)

    def get_final_embeddings(self, edge_index, edge_weight=None):
        self.eval()
        with torch.no_grad():
            x = self.forward(edge_index, edge_weight)
        return x

def bpr_loss(emb, pos_src, pos_dst, neg_dst, l2_lambda=1e-4):
    src_emb = emb[pos_src]
    pos_emb = emb[pos_dst]
    neg_emb = emb[neg_dst]
    pos_scores = (src_emb * pos_emb).sum(dim=1)
    neg_scores = (src_emb * neg_emb).sum(dim=1)
    loss = -F.logsigmoid(pos_scores - neg_scores).mean()
    reg_loss = src_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)
    return loss + l2_lambda * reg_loss / src_emb.size(0)



def sample_negatives(pos_src, num_nodes, existing_edges):
    neg = []
    for s in pos_src:
        neg_i = np.random.randint(num_nodes)
        while ((int(s), neg_i) in existing_edges) or (neg_i == s):
            neg_i = np.random.randint(num_nodes)
        neg.append(neg_i)
    return torch.tensor(neg, dtype=torch.long)