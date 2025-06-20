import torch
import torch.nn as nn
import torch.nn.functional as F
class FusionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, act, attr):
        x = torch.cat([act, attr], dim=1)
        return self.net(x)
    
def bpr_loss(user_emb, pos_emb, neg_emb):
    pos_scores = (user_emb * pos_emb).sum(dim=1)
    neg_scores = (user_emb * neg_emb).sum(dim=1)
    return -F.logsigmoid(pos_scores - neg_scores).mean()