import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

Top_k = 20
user_emb_path = "/kaggle/input/final_embeddings/pytorch/default/1/final_user_embeddings.pt"
item_emb_path = "/kaggle/input/final_embeddings/pytorch/default/1/final_item_embeddings.pt"
val_path = "/kaggle/input/movie-len/validation.csv"


# 计算recall@k
def recall_at_k(preds, truth, k):
    hits = len(set(preds[:k]) & truth)
    return hits / len(truth) if truth else 0.0

# precision@k
def precision_at_k(preds, ground_truth, k):
    hits = len(set(preds[:k]) & ground_truth)
    return hits / k

# ndcg@k
def ndcg_at_k(preds, truth, k):
    dcg = 0.0
    for i, p in enumerate(preds[:k]):
        if p in truth:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(truth), k)))
    return dcg / idcg if idcg > 0 else 0.0

# hit_rate@k
def hit_rate_at_k(preds, truth, k):
    return 1.0 if len(set(preds[:k]) & truth) > 0 else 0.0

# 加载数据
user_emb = torch.load(user_emb_path, map_location='cpu')
item_emb = torch.load(item_emb_path, map_location='cpu')
val_df = pd.read_csv(val_path)
print("成功加载数据")

user_emb = user_emb / user_emb.norm(dim=1, keepdim=True)
item_emb = item_emb / item_emb.norm(dim=1, keepdim=True)

# 获取用户真实交互
val_dict = val_df.groupby('userId')['itemId'].apply(set).to_dict()
num_users = user_emb.shape[0]
print("成功获取用户真实交互")


# 进行推荐
recalls, precisions, ndcgs, hits = [], [], [], []

users = tqdm(range(num_users), desc="Evaluating users")
for u in users:
    # 获取用户交互的物品
    if u not in val_dict:
        continue
    true_items = val_dict[u]
    if not true_items:
        continue

    scores = torch.matmul(user_emb[u], item_emb.T) # 点乘得到分数
    topk_items = torch.topk(scores, Top_k).indices.tolist() # 获取推荐列表

    recalls.append(recall_at_k(topk_items, true_items, Top_k))
    precisions.append(precision_at_k(topk_items, true_items, Top_k))
    ndcgs.append(ndcg_at_k(topk_items, true_items, Top_k))
    hits.append(hit_rate_at_k(topk_items, true_items, Top_k))

print("推荐结果指标：")
print(f"Recall@{Top_k}   = {np.mean(recalls):.4f}")
print(f"Precision@{Top_k}   = {np.mean(precisions):.4f}")
print(f"NDCG@{Top_k}     = {np.mean(ndcgs):.4f}")
print(f"HitRate@{Top_k}  = {np.mean(hits):.4f}")

