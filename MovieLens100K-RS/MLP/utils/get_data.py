import torch
import pandas as pd
def get_data():
    action_emb = torch.load("/kaggle/input/embedding/pytorch/default/1/item_action_embeddings.pt")
    attr_emb = torch.load("/kaggle/input/embedding/pytorch/default/1/attribute_embeddings.pt")

    assert attr_emb.shape == action_emb.shape # 保证数量一致
    num_items, emb_dim = attr_emb.shape

    # 交互数据
    df = pd.read_csv("/kaggle/input/movie-len/train.csv")
    user_item_dict = df.groupby('userId')['itemId'].apply(list).to_dict()
    user_ids = sorted(user_item_dict.keys())
    num_users = max(user_ids) + 1

    return num_items, emb_dim, action_emb, attr_emb, num_users, user_item_dict