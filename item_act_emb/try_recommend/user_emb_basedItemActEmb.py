import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random
import itertools
from item_act_emb.get_grapgh import get_count
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

# 加载item嵌入
train_csv = "/kaggle/input/movielens-1m-rs/train.csv"
item_emb_path = "/kaggle/working/item_act_embeddings.pt"
item_emb = torch.load(item_emb_path, map_location='cpu', weights_only=False) 
num_users, num_items = get_count() 
emb_dim = item_emb.shape[1]
print("num item:", num_items)
print("num user:", num_users)
print("emb_dim:", emb_dim)

# 构建用户和物品字典 uerIdx:[itemIdx1, itemIdx2, itemIdx3, ... ]
df = pd.read_csv(train_csv)
df = df[df['score'] >= 4]
user_item_dict = df.groupby('userIdx')['itemIdx'].apply(list).to_dict()



device = 'cuda' if torch.cuda.is_available() else 'cpu'

item_emb = item_emb.to(device)

user_pooling = UserAttentionPooling(emb_dim).to(device)


epochs = 10
batch_size = 1024 # 每轮处理多少个采样
learning_rate = 0.001
num_samples = 50000 

optimizer = torch.optim.Adam(user_pooling.parameters(), lr=learning_rate, weight_decay=1e-4)


for epoch in range(epochs):
    user_pooling.train()
    users, pos_items, neg_items = user_sample_triplets(user_item_dict, num_items, num_samples=num_samples)

    epoch_loss = 0
    loop = tqdm(range(0, len(users), batch_size), desc=f"Epoch {epoch+1}/{epochs}")
    for start in loop:
        end = start + batch_size
        u_batch = users[start:end].to(device)
        i_batch = pos_items[start:end].to(device)
        j_batch = neg_items[start:end].to(device)

        emb_i = item_emb[i_batch]
        emb_j = item_emb[j_batch]

        emb_user = []
        his_embs_list = []
        lens = []
        for u in u_batch.cpu().tolist():
            item_list = user_item_dict.get(u, [])
            if not item_list:
                his_embs_list.append(torch.zeros(1,emb_dim, device=device))
                lens.append(1)
            else:
                emb = item_emb[item_list]
                his_embs_list.append(emb)
                lens.append(len(item_list))
        all_his_embs = torch.cat(his_embs_list, dim=0) 
        offsets = [0] + list(itertools.accumulate(lens))

        emb_user = []
        for i in range(len(u_batch)):
            start, end = offsets[i], offsets[i+1]
            user_emb = user_pooling(all_his_embs[start:end])
            emb_user.append(user_emb)
        emb_user = torch.stack(emb_user)

        loss = bpr_loss(emb_user, emb_i, emb_j)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    epoch_loss = epoch_loss / len(loop)

    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

# 保存最终嵌入
user_pooling.eval()
final_user_emb = []
with torch.no_grad():
    for u in range(num_users):
        item_list = user_item_dict.get(u, [])
        if not item_list:
            final_user_emb.append(torch.zeros(item_emb.shape[1]))
        else:
            his_embs = item_emb[item_list]
            user_emb = user_pooling(his_embs)
            final_user_emb.append(user_emb.cpu())
final_user_emb = torch.stack(final_user_emb)
print("user emb 10: ", final_user_emb[:10])
torch.save(final_user_emb, "user_embeddings.pt")
print("最终嵌入保存完毕！")

# user_emb = torch.load("user_embeddings.pt")