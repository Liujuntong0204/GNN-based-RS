import torch
import numpy as np
def sample_triplets(user_item_dict, num_items, num_samples=10000):
    users, pos_items, neg_items = [], [], []
    for _ in range(num_samples):
        u = np.random.choice(list(user_item_dict.keys()))
        pos_list = user_item_dict[u]
        if not pos_list:
            continue
        i = np.random.choice(pos_list)
        j = np.random.randint(num_items)
        while j in pos_list:
            j = np.random.randint(num_items)
        users.append(u)
        pos_items.append(i)
        neg_items.append(j)
    return torch.LongTensor(users), torch.LongTensor(pos_items), torch.LongTensor(neg_items)