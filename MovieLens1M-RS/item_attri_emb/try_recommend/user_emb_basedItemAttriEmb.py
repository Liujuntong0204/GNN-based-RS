import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import random
from get_graph import get_item_features, get_gragh
from PDA_GNN import PDA_GNN

class UserAttentionPooling(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.query_vector = nn.Parameter(torch.randn(emb_dim))
    def forward(self, his_embs, user_indices):
        user_embs = []
        for i in range(len(user_indices) - 1):
            start, end = user_indices[i], user_indices[i + 1]
            user_hist = his_embs[start:end]  # [len, emb_dim]
            scores = torch.matmul(user_hist, self.query_vector)
            weights = F.softmax(scores, dim=0)
            user_emb = (weights.unsqueeze(1) * user_hist).sum(dim=0)
            user_embs.append(user_emb)
        return torch.stack(user_embs)

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

# 训练

item_count, data_title, data_year, data_cat = get_gragh()

init_feat = get_item_features()
print("初始特征 shape:", init_feat.shape)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

title_num_layers = 2
year_num_layers = 1
cat_num_layers = 1

model = PDA_GNN(
    init_feat=init_feat,
    num_layers_title=title_num_layers,
    num_layers_year=year_num_layers,
    num_layers_cat=cat_num_layers
).to(device)

user_pooling = UserAttentionPooling(64).to(device)

train_csv = "/kaggle/input/movielens-1m-rs/train.csv"
df = pd.read_csv(train_csv)
df = df[df['score'] >= 4]  
user_item_dict = df.groupby('userIdx')['itemIdx'].apply(list).to_dict()

def freeze_gnn_layers(model):
    for param in model.convs_title.parameters():
        param.requires_grad = False
    for param in model.convs_year.parameters():
        param.requires_grad = False
    for param in model.convs_cat.parameters():
        param.requires_grad = False

# epochs = 20
stage1_epochs = 5   # 第一阶段：训练所有模块
stage2_epochs = 20  # 第二阶段：冻结GNN，仅训练注意力融合
batch_size = 1024
num_samples = 50000
lr = 0.001

# 第一阶段 传播+两个注意力权重学习
optimizer = optim.Adam(list(model.parameters()) + list(user_pooling.parameters()), lr=lr, weight_decay=1e-4)

for epoch in range(stage1_epochs):
    model.train()
    user_pooling.train()

    users, pos_items, neg_items = user_sample_triplets(user_item_dict, item_count, num_samples)

    epoch_loss = 0
    loop = tqdm(range(0, len(users), batch_size), desc=f"Epoch {epoch + 1}/{stage1_epochs}")

    full_item_emb = model(data_title.edge_index.to(device),
                            data_year.edge_index.to(device),
                            data_cat.edge_index.to(device))  # [num_items, emb_dim]

    for start in loop:
        end = start + batch_size
        u_batch = users[start:end].to(device)
        i_batch = pos_items[start:end].to(device)
        j_batch = neg_items[start:end].to(device)

        full_item_emb = model(data_title.edge_index.to(device),
                            data_year.edge_index.to(device),
                            data_cat.edge_index.to(device))

        emb_i = full_item_emb[i_batch]
        emb_j = full_item_emb[j_batch]

        his_embs_list = []
        lens = []
        for u in u_batch.cpu().tolist():
            items_his = user_item_dict.get(u, [])
            if not items_his:
                his_embs_list.append(torch.zeros(1, full_item_emb.shape[1], device=device))
                lens.append(1)
            else:
                his_embs_list.append(full_item_emb[items_his])
                lens.append(len(items_his))

        all_his_embs = torch.cat(his_embs_list, dim=0)
        offsets = [0] + list(itertools.accumulate(lens))
        offsets = torch.LongTensor(offsets).to(device)
        
        emb_user = user_pooling(all_his_embs, offsets)

        loss = bpr_loss(emb_user, emb_i, emb_j)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(loop):.4f}")


# 第二阶段 不传播了 只训练注意力权重
freeze_gnn_layers(model)
optimizer = optim.Adam(
    list(model.att_mlp.parameters()) + list(user_pooling.parameters()),
    lr=lr, weight_decay=1e-4
)
for epoch in range(stage2_epochs):
    model.train()
    user_pooling.train()

    users, pos_items, neg_items = user_sample_triplets(user_item_dict, item_count, num_samples)

    epoch_loss = 0
    loop = tqdm(range(0, len(users), batch_size), desc=f"Epoch {epoch + 1}/{stage2_epochs}")

    full_item_emb = model(data_title.edge_index.to(device),
                            data_year.edge_index.to(device),
                            data_cat.edge_index.to(device))  # [num_items, emb_dim]

    for start in loop:
        end = start + batch_size
        u_batch = users[start:end].to(device)
        i_batch = pos_items[start:end].to(device)
        j_batch = neg_items[start:end].to(device)

        full_item_emb = model(data_title.edge_index.to(device),
                            data_year.edge_index.to(device),
                            data_cat.edge_index.to(device))

        emb_i = full_item_emb[i_batch]
        emb_j = full_item_emb[j_batch]

        his_embs_list = []
        lens = []
        for u in u_batch.cpu().tolist():
            items_his = user_item_dict.get(u, [])
            if not items_his:
                his_embs_list.append(torch.zeros(1, full_item_emb.shape[1], device=device))
                lens.append(1)
            else:
                his_embs_list.append(full_item_emb[items_his])
                lens.append(len(items_his))

        all_his_embs = torch.cat(his_embs_list, dim=0)
        offsets = [0] + list(itertools.accumulate(lens))
        offsets = torch.LongTensor(offsets).to(device)
        
        emb_user = user_pooling(all_his_embs, offsets)

        loss = bpr_loss(emb_user, emb_i, emb_j)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(loop):.4f}")

print("训练完成")


def user_pooling_single(his_embs, user_pooling):
    scores = torch.matmul(his_embs, user_pooling.query_vector)
    weights = F.softmax(scores, dim=0)
    user_emb = (weights.unsqueeze(1) * his_embs).sum(dim=0)  
    return user_emb

model.eval()
user_pooling.eval()
with torch.no_grad():
    final_item_emb = model(data_title.edge_index.to(device),
                            data_year.edge_index.to(device),
                            data_cat.edge_index.to(device))
    final_user_emb = []
    for u in range(max(user_item_dict.keys()) + 1):
        items_his = user_item_dict.get(u, [])
        if not items_his:
            print("no items")
            final_user_emb.append(torch.zeros(final_item_emb.shape[1], device=device))
        else:
            his_embs = final_item_emb[items_his].to(device)
            user_emb = user_pooling_single(his_embs, user_pooling)
            final_user_emb.append(user_emb)
    final_user_emb = torch.stack(final_user_emb)

print("前十个物品嵌入:\n", final_item_emb[:10])
print(final_item_emb.shape)
print("前十个用户嵌入:\n", final_user_emb[:10])
print(final_user_emb.shape)
torch.save(final_item_emb.cpu(), "item_attribute_embeddings.pt")
torch.save(final_user_emb.cpu(), "user_embeddings.pt")
print("训练好的属性嵌入和用户嵌入已保存")