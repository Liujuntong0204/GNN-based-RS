import torch
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
from item_act_emb.get_grapgh import get_action_graph
from item_act_emb.LightGCN import LightGCN, sample_negatives, bpr_loss

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 参数
learning_rate = 0.001
embedding_dim = 64
num_layers = 3
epochs = 10
batch_size = 1024 # 采样的中心节点数
num_neighbors = [10,10,10]
l2_lambda = 1e-4
dropout = 0.5
k = 100 # 每个节点最多邻居数

train_csv_path = "/kaggle/input/movielens-1m-rs/train.csv"
item_count, data = get_action_graph(train_csv_path, k = k)

edge_index = data.edge_index
edge_weight = data.edge_weight

loader = NeighborLoader(
    data=data,
    num_neighbors=num_neighbors,
    batch_size=batch_size,
    shuffle=True,
)


model = LightGCN(num_nodes=item_count, embedding_dim=embedding_dim, num_layers=num_layers, dropout=dropout).to(device)
opt = torch.optim.Adam(model.parameters(), lr = learning_rate,  weight_decay=1e-4)

existing = set((int(s), int(d)) for s, d in edge_index.t().tolist())

model.train()
for epoch in range(epochs):
    epoch_loss = 0
    loop = tqdm(loader, total=len(loader), desc=f"Epoch {epoch + 1}/{epochs}")
    for batch in loop :
        batch = batch.to(device)  
        
        emb = model(edge_index=batch.edge_index, edge_weight=batch.edge_weight) 

        src, dst = batch.edge_index
        src_global = batch.n_id[src] # 这里要获取全局的索引
        dst_global = batch.n_id[dst]

        neg_global = sample_negatives(src_global.cpu(), item_count, existing)

        loss = bpr_loss(emb, src_global.to(device), dst_global.to(device), neg_global.to(device), l2_lambda=l2_lambda)
        
        opt.zero_grad()
        loss.backward()
        opt.step()

        epoch_loss += loss.item()

        loop.set_postfix(loss=loss.item())

    epoch_loss /= len(loader)
    print(f'Epoch {epoch + 1}, Loss {epoch_loss:.4f}')

with torch.no_grad():
    final_embeddings = model.get_final_embeddings(edge_index.to(device), edge_weight.to(device))
    torch.save(final_embeddings, "item_act_embeddings.pt")
    print("Item embeddings saved.")
    print("Final embedding shape:", final_embeddings.shape)
    print("Example embedding for itemIdx 0-4:", final_embeddings[:5])