import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# recall@k
def recall_at_k(preds, truth, k):
    hits = len(set(preds[:k]) & truth)
    return hits / len(truth) if truth else 0.0

# precision@k
def precision_at_k(preds, truth, k):
    hits = len(set(preds[:k]) & truth)
    return hits / k

# ndcg@k
def ndcg_at_k(preds, truth, k):
    dcg = sum(1.0 / np.log2(i + 2) for i, p in enumerate(preds[:k]) if p in truth)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(truth), k)))
    return dcg / idcg if idcg > 0 else 0.0

# hit_rate@k
def hit_rate_at_k(preds, truth, k):
    return 1.0 if len(set(preds[:k]) & truth) > 0 else 0.0



train_df = pd.read_csv("/kaggle/input/movielens-1m-rs/train.csv")
train_user_item_dict = train_df.groupby("userIdx")["itemIdx"].apply(set).to_dict()

item_emb_path = "/kaggle/working/item_act_embeddings.pt"
item_emb = torch.load(item_emb_path, map_location='cpu', weights_only=False)  

user_emb_path = "/kaggle/working/user_embeddings.pt"
user_emb = torch.load(user_emb_path, map_location='cpu', weights_only=False)           
    
val_path = "/kaggle/input/movielens-1m-rs/validation.csv"
Top_k = 10


user_emb = user_emb / user_emb.norm(dim=1, keepdim=True)
item_emb = item_emb / item_emb.norm(dim=1, keepdim=True)

# 获取用户真实交互
val_df = pd.read_csv(val_path)
val_dict = val_df.groupby('userIdx')['itemIdx'].apply(set).to_dict()
for user, items in list(val_dict.items())[:10]:
    print(f"用户 {user} 的验证集中真实交互电影：{items}")
print("成功获取用户真实交互")


# 进行推荐
recalls, precisions, ndcgs, hits = [], [], [], []
val_user_idx = val_df['userIdx'].unique()
print("验证集中用户数量：", len(val_user_idx))

users = tqdm(val_user_idx, desc="Evaluating")
for user_idx in users:
    scores = torch.matmul(user_emb[user_idx], item_emb.T)   # 计算分数
    seen_items = train_user_item_dict.get(user_idx, set()) # 获取用户已经看过的集合
    if seen_items:
        scores[list(seen_items)] = float('-inf') # 看过的电影的评分设为-inf
        
    topk_items = torch.topk(scores, Top_k).indices.tolist() # 得到推荐列表 这里！应该推荐他没看过的电影
    true_items = val_dict.get(user_idx, set())

    recalls.append(recall_at_k(topk_items, true_items, Top_k))
    precisions.append(precision_at_k(topk_items, true_items, Top_k))
    ndcgs.append(ndcg_at_k(topk_items, true_items, Top_k))
    hits.append(hit_rate_at_k(topk_items, true_items, Top_k))

print("推荐结果指标：")
print(f"Recall@{Top_k}   = {np.mean(recalls):.4f}")
print(f"Precision@{Top_k}   = {np.mean(precisions):.4f}")
print(f"NDCG@{Top_k}     = {np.mean(ndcgs):.4f}")
print(f"HitRate@{Top_k}  = {np.mean(hits):.4f}")


"""
推荐结果指标：
Recall@10   = 0.0410
Precision@10   = 0.1075
NDCG@10     = 0.1157
HitRate@10  = 0.4886
"""