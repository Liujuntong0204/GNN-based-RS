import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'

item_act_emb_path = "/kaggle/input/item-embedding/item_act_embeddings.pt"
item_attri_emb_path = "/kaggle/input/item-embedding/item_attribute_embeddings.pt"

item_act_emb = torch.load(item_act_emb_path, map_location='cpu', weights_only=False)
print("item_act_emb shape", item_act_emb.shape)

item_attri_emb = torch.load(item_attri_emb_path, map_location='cpu', weights_only=False)  
print("item_attri_emb shape", item_attri_emb.shape)

item_act_emb = item_act_emb.to(device)
item_attri_emb = item_attri_emb.to(device)

item_emb = torch.cat([item_act_emb, item_attri_emb], dim=1)  # [num_items, 128]
print("item_final_emb shape", item_emb.shape)

torch.save(item_emb, "item_embeddings.pt")