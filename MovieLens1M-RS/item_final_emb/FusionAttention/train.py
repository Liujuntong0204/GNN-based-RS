import itertools
import pandas as pd
import torch
from tqdm import tqdm
import itertools
from FusionAttention import FusionAttention, get_count, UserAttentionPooling,  user_sample_triplets, bpr_loss


# 加载item嵌入和映射
train_csv = "/kaggle/input/movielens-1m-rs/train.csv"

num_users, num_items = get_count() 

print("num item:", num_items)
print("num user:", num_users)


# 构建用户和物品字典 uerIdx:[itemIdx1, itemIdx2, itemIdx3, ... ]
df = pd.read_csv(train_csv)
df = df[df['score'] >= 4]
user_item_dict = df.groupby('userIdx')['itemIdx'].apply(list).to_dict()




device = 'cuda' if torch.cuda.is_available() else 'cpu'

item_act_emb_path = "/kaggle/input/item-embedding/item_act_embeddings.pt"
item_act_emb = torch.load(item_act_emb_path, map_location='cpu', weights_only=False)
item_act_emb = item_act_emb.to(device)

item_attri_emb_path = "/kaggle/input/item-embedding/item_attribute_embeddings.pt"
item_attri_emb = torch.load(item_attri_emb_path, map_location='cpu', weights_only=False)
item_attri_emb = item_attri_emb.to(device)


fusion_model = FusionAttention(emb_dim=item_act_emb.shape[1]).to(device)

emb_dim = fusion_model.emb_dim

user_pooling = UserAttentionPooling(emb_dim).to(device)



epochs = 20
batch_size = 1024 # 每轮处理多少个采样
learning_rate = 0.0001
num_samples = 50000 

optimizer = torch.optim.Adam(
    list(user_pooling.parameters()) + list(fusion_model.parameters()),
    lr=learning_rate,
    weight_decay=1e-4
)

user_item_tensor_dict = {
    u: torch.LongTensor(item_list).to(device)
    for u, item_list in user_item_dict.items()
}

for epoch in range(epochs):
    user_pooling.train()
    fusion_model.train()
    users, pos_items, neg_items = user_sample_triplets(user_item_dict, num_items, num_samples=num_samples)

    epoch_loss = 0
    # 融合
    loop = tqdm(range(0, len(users), batch_size), desc=f"Epoch {epoch+1}/{epochs}")
    for start in loop:
        end = start + batch_size
        u_batch = users[start:end].to(device)
        i_batch = pos_items[start:end].to(device)
        j_batch = neg_items[start:end].to(device)

        i_act = item_act_emb[i_batch]
        i_attr = item_attri_emb[i_batch]
        emb_i = fusion_model(i_act, i_attr)

        j_act = item_act_emb[j_batch]
        j_attr = item_attri_emb[j_batch]
        emb_j = fusion_model(j_act, j_attr)
        # emb_i = item_emb[i_batch]
        # emb_j = item_emb[j_batch]

        his_embs_list = []
        lens = []
        for u in u_batch.cpu().tolist():
            his_items = user_item_tensor_dict.get(u, None)
            if his_items is None or len(his_items) == 0:
                his_embs_list.append(torch.zeros(1, emb_dim, device=device))
                lens.append(1)
            else:
                act = item_act_emb[his_items]
                attr = item_attri_emb[his_items]
                emb = fusion_model(act, attr)
                his_embs_list.append(emb)
                lens.append(len(his_items))
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
fusion_model.eval()
with torch.no_grad():
    item_emb = fusion_model(item_act_emb, item_attri_emb)
torch.save(item_emb, "item_embeddings.pt")
# print("item_emb shape: ", item_emb.shape)

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
# print("user emb 10: ", final_user_emb[:10])
# print(final_user_emb.shape)
torch.save(final_user_emb, "user_embeddings.pt")
# print("最终嵌入保存完毕！")

# user_emb = torch.load("user_embeddings.pt")
