# 获取图
import pandas as pd
from collections import defaultdict
import numpy as np
import torch
from torch_geometric.data import Data

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


def get_action_graph(train_csv_path, k=50):
    df = pd.read_csv(train_csv_path)
    df = df[df['score'] >= 4]

    user_count, item_count = get_count() 

    print("item_count: ", item_count)
    print("user_count: ", user_count)

    assert 'userIdx' in df.columns and 'itemIdx' in df.columns
    assert df['userIdx'].isnull().sum() == 0
    assert df['itemIdx'].isnull().sum() == 0
    
    grouped = df.groupby('userIdx')['itemIdx'].apply(list)
    print("grouped 10:", grouped[:10])

    item_c = defaultdict(lambda: defaultdict(int))

    for item_list in grouped:
        for i in range(len(item_list)):
            for j in range(i + 1, len(item_list)):
                a, b = item_list[i], item_list[j]
                item_c[a][b] += 1
                item_c[b][a] += 1

    edge_list = []
    for i, neighbors in item_c.items():
        topk_neighbors = sorted(neighbors.items(), key=lambda x: -x[1])[:k]
        for j, weight in topk_neighbors:
            edge_list.append((i, j, np.log1p(weight)))
    print("egd_count: ", len(edge_list))
    print("edges 10条: ", edge_list[:10])
    edge_index = torch.tensor([[s, t] for s, t, _ in edge_list], dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor([w for _, _, w in edge_list], dtype=torch.float)
    data = Data(edge_index=edge_index, edge_weight=edge_weight, num_nodes=item_count)
    return item_count, data