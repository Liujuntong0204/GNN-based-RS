import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import LGConv


class LightGCN(nn.Module):
    def __init__(self, num_nodes, embedding_dim = 64, num_layers = 2):
        super(LightGCN, self).__init__()

        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

        self.convs = nn.ModuleList([LGConv() for _ in range(num_layers)])

    def forward(self, x=None, edge_index=None):
        if x is None:
            x = self.embedding.weight
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs.append(x)

        return sum(xs) / len(xs)

    def get_embeddings(self):
        return self.embedding.weight.detach()

def bpr_loss(emb, pos_indices, neg_indices):
    i, j = pos_indices
    k_i, k = neg_indices
    
    pos_score = (emb[i] * emb[j]).sum(1)
    neg_score = (emb[k_i] * emb[k]).sum(1)

    return -F.logsigmoid(pos_score - neg_score).mean()