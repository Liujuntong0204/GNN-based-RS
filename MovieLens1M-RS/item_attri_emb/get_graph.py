
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer, util

# 获取初始嵌入
def get_item_features():
    items = pd.read_csv("/kaggle/input/movielens-1m-rs/items.csv")
    category_columns = items.columns[4:]

    # Title
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    title_emb = model.encode(items['Title'].tolist(), convert_to_tensor=True).cpu()  # [N, D]

    # Category 
    category_emb = torch.tensor(items[category_columns].values, dtype=torch.float32)  # [N, C]

    # 拼接再映射为64维
    concat_feat = torch.cat([title_emb, category_emb], dim=1)
    feat_dim = concat_feat.shape[1]
    projection = nn.Linear(feat_dim, 64)
    with torch.no_grad():
        init_feat = F.normalize(projection(concat_feat), dim=1)
    return init_feat


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

def get_gragh():
    import random
    items = pd.read_csv("/kaggle/input/movielens-1m-rs/items.csv")
    item_count = len(items)
    print("item_count: ", item_count)

    item_idxs = items['itemIdx'].tolist()
    idx_map = {i: item_idxs[i] for i in range(item_count)}  # 索引到 itemIdx

    ## Title Top-K >=0.5
    title_topk=10
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    title_embs = model.encode(items['Title'].tolist(), convert_to_tensor=True)
    print("已获取title向量")

    sim_matrix = util.cos_sim(title_embs, title_embs).cpu()  
    edges = set()
    for i in range(item_count):
        sims = sim_matrix[i]
        sims[i] = -1  # 排除自己
        topk = torch.topk(sims, k=title_topk)
        for j, sim in zip(topk.indices, topk.values):
            if sim >= 0.5:
                edges.add((idx_map[i], idx_map[j.item()]))
    print("title连接边数（单向）:", len(edges))
    # 双向
    src, dst = zip(*edges)
    edges_title = torch.tensor([list(src) + list(dst), list(dst) + list(src)])
    title_Data = Data(edge_index=edges_title, num_nodes=item_count)

    ## Year 同年份内随机连接 10 个
    years = items['Year']
    year_groups = items.groupby('Year')['itemIdx'].apply(list).to_dict()
    edges = set()
    for year, item_list in year_groups.items():
        if len(item_list) <= 1:
            continue
        for i in item_list:
            candidates = [j for j in item_list if j != i]
            sampled = random.sample(candidates, min(len(candidates), 10))
            for j in sampled:
                edges.add((i, j))
    print("year连接边数（单向）:", len(edges))
    src, dst = zip(*edges)
    edges_year = torch.tensor([list(src) + list(dst), list(dst) + list(src)])
    year_Data = Data(edge_index=edges_year, num_nodes=item_count)

    ## Category 共类数量 Top-K（至少 1 个
    cat_topk=10
    category_columns = items.columns[4:]
    cat_matrix = torch.tensor(items[category_columns].values, dtype=torch.float32)  
    common_cat = torch.matmul(cat_matrix, cat_matrix.T)  
    edges = set()
    for i in range(item_count):
        sims = common_cat[i]
        sims[i] = -1  # 排除自己
        topk = torch.topk(sims, k=cat_topk)
        for j, overlap in zip(topk.indices, topk.values):
            if overlap >= 1:
                edges.add((idx_map[i], idx_map[j.item()]))
    print("category连接边数（单向）:", len(edges))
    src, dst = zip(*edges)
    edges_cat = torch.tensor([list(src) + list(dst), list(dst) + list(src)])
    categories_Data = Data(edge_index=edges_cat, num_nodes=item_count)

    return item_count, title_Data, year_Data, categories_Data