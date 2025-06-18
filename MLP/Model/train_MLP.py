import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.get_data import get_data
from Model.MLP import FusionMLP, bpr_loss
from utils.sampler import sample_triplets

num_items, emb_dim, action_emb, attr_emb, num_users, user_item_dict = get_data()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
fusion_model = FusionMLP(input_dim=2*emb_dim, output_dim=emb_dim).to(device)

# 参数
epochs = 10
batch_size = 1024 # 一次学习采样中的用户数
learning_rate = 0.001
num_samples = 50000 

optimizer = torch.optim.Adam(fusion_model.parameters(), lr=learning_rate)

action_emb = action_emb.to(device)
attr_emb = attr_emb.to(device)

for epoch in range(epochs):
    fusion_model.train()
    users, pos_items, neg_items = sample_triplets(user_item_dict, num_items, num_samples=num_samples)
    
    epoch_loss = 0
    loop = tqdm(range(0, len(users), batch_size), desc=f"Epoch {epoch+1}/{epochs}")
    for start in loop:
        end = start + batch_size
        u_batch = users[start:end].to(device)
        i_batch = pos_items[start:end].to(device)
        j_batch = neg_items[start:end].to(device)

        
        emb_i_act, emb_j_act = action_emb[i_batch], action_emb[j_batch]
        emb_i_attr, emb_j_attr = attr_emb[i_batch], attr_emb[j_batch]

        emb_i = fusion_model( emb_i_act, emb_i_attr)
        emb_j = fusion_model( emb_j_act, emb_j_attr)

        # 构建用户嵌入（通过交互物品平均）
        emb_user = []
        for u in u_batch.cpu().tolist():
            item_list = user_item_dict[u]
            act = action_emb[item_list]
            attr = attr_emb[item_list]
            emb = fusion_model(act, attr).mean(dim=0)
            emb_user.append(emb)
        emb_user = torch.stack(emb_user).to(device)

        loss = bpr_loss(emb_user, emb_i, emb_j)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
    if (epoch + 1) % 5 == 0:
        save_path = f"fusion_model_epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': fusion_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss
        }, save_path)
        print(f"model saved to {save_path}")

fusion_model.eval()
with torch.no_grad():
    final_item_emb = fusion_model(action_emb, attr_emb).cpu()

    # 生成用户嵌入（使用最终物品嵌入）
    final_user_emb = []
    for u in range(num_users):
        item_list = user_item_dict.get(u, [])
        if not item_list:
            final_user_emb.append(torch.zeros_like(final_item_emb[0]))
        else:
            final_user_emb.append(final_item_emb[item_list].mean(dim=0))
    final_user_emb = torch.stack(final_user_emb)

torch.save(final_item_emb, "final_item_embeddings.pt")
torch.save(final_user_emb, "final_user_embeddings.pt")
print("最终嵌入保存完毕！")