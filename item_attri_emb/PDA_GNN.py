import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import LGConv

class PDA_GNN(nn.Module):
    def __init__(self, init_feat, num_layers_title=2, num_layers_year=1, num_layers_cat=3,hidden_dim=64):
        super(PDA_GNN, self).__init__()
        self.init_feat = init_feat

        # 构建三个图的卷积层
        self.convs_title = nn.ModuleList([LGConv() for _ in range(num_layers_title)])
        self.convs_year = nn.ModuleList([LGConv() for _ in range(num_layers_year)])
        self.convs_cat = nn.ModuleList([LGConv() for _ in range(num_layers_cat)])

        # 注意力机制
        self.att_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 3个attention分数
        )

    def forward(self, edge_index_title, edge_index_year, edge_index_cat):
        x = self.init_feat.to(edge_index_title.device)
        
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

        
        x_all = torch.cat([x_title, x_year, x_cat], dim=1)

        att_logits = self.att_mlp(x_all)
        att_weights = F.softmax(att_logits, dim=1)  # [N, 3]

        # 融合
        out = (
            att_weights[:, 0:1] * x_title +
            att_weights[:, 1:2] * x_year +
            att_weights[:, 2:3] * x_cat
        )
        return out

    def get_final_embeddings(self, edge_index_title, edge_index_year, edge_index_cat):
        self.eval()
        with torch.no_grad():
            return self.forward(edge_index_title, edge_index_year, edge_index_cat)
        