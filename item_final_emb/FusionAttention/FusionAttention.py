import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
from torch_geometric.nn.conv import LGConv
from tqdm import tqdm
import random
import itertools
from collections import defaultdict

class FusionAttention(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.linear = nn.Linear(emb_dim * 2, 2)  # 用于计算两个嵌入的权重
        self.emb_dim = emb_dim

    def forward(self, act, attr):
        # act, attr: [batch_size, emb_dim]
        combined = torch.cat([act, attr], dim=1)  # [batch_size, 2*emb_dim]
        weights = F.softmax(self.linear(combined), dim=1)  # [batch_size, 2]
        w_act = weights[:, 0].unsqueeze(1)  # [batch_size, 1]
        w_attr = weights[:, 1].unsqueeze(1)
        fused = w_act * act + w_attr * attr  # 加权和
        return fused  # [batch_size, emb_dim]


def get_count():
    import pickle
    mapping_dict = {}
    with open("/kaggle/input/id-idx-mapping/id_mappings.pkl", "rb") as f:
        mapping_dict = pickle.load(f)
    user_id2idx = mapping_dict["user_id2idx"]
    user_count = len(user_id2idx)
    item_id2idx = mapping_dict["movie_id2idx"]
    item_count = len(item_id2idx)
    # print("user num: ", user_count)
    # print("item num: ", item_count)
    return user_count, item_count


class UserAttentionPooling(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.query_vector = nn.Parameter(torch.randn(emb_dim))
    def forward(self, emb_items):
        scores = torch.matmul(emb_items, self.query_vector)  
        weights = F.softmax(scores, dim=0)                  
        user_emb = (weights.unsqueeze(1) * emb_items).sum(dim=0)
        return user_emb 

def user_sample_triplets(user_item_dict, num_items, num_samples=10000):
    users, pos_items, neg_items = [], [], []
    for _ in range(num_samples):
        u = np.random.choice(list(user_item_dict.keys()))
        pos = random.choice(user_item_dict[u])
        neg = random.randint(0, num_items - 1)
        while (neg in user_item_dict[u]) or (neg==pos):
            neg = random.randint(0, num_items - 1)
        users.append(u)
        pos_items.append(pos)
        neg_items.append(neg)
    return torch.LongTensor(users), torch.LongTensor(pos_items), torch.LongTensor(neg_items)

def bpr_loss(user_emb, pos_emb, neg_emb, l2_lambda=1e-4):
    pos_scores = (user_emb * pos_emb).sum(dim=1)
    neg_scores = (user_emb * neg_emb).sum(dim=1)
    loss = -F.logsigmoid(pos_scores - neg_scores).mean()
    reg = user_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)
    return loss + l2_lambda * reg / user_emb.size(0)
