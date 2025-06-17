import torch
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
from tqdm import tqdm
from utils.item_action_graph import get_action_graph
from Model.lightGCN import LightGCN, bpr_loss
from utils.sampler import sample_negatives


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 参数
learning_rate = 0.01
embedding_dim = 64
num_layers = 2
epochs = 10
batch_size = 10
num_neighbors = [10,10]

item_count, edges = get_action_graph()

edge_index = torch.tensor(edges[['source', 'target']].values, dtype=torch.long).t().contiguous() # [2, E]
data = Data(edge_index=edge_index, num_nodes=item_count)

loader = NeighborLoader(
    data=data,
    num_neighbors=num_neighbors,
    batch_size=batch_size,
    shuffle=True,
    input_nodes=None
)


model = LightGCN(num_nodes=item_count, embedding_dim=embedding_dim, num_layers=num_layers).to(device)
opt = torch.optim.Adam(model.parameters(), lr = learning_rate)

existing = set([tuple([s.item(), d.item()]) for s, d in edge_index.t()])
nodes = range(item_count)

model.train()
for epoch in range(epochs):
    epoch_loss = 0
    loop = tqdm(loader, total=len(loader), desc=f"Epoch {epoch + 1}/{epochs}")
    for batch in loop :
        batch = batch.to(device)  # 这句话非常重要，确保数据在 GPU
        src, dst = batch.edge_index
        
        emb = model(None, batch.edge_index)  # 前向传播时输入 GPU数据

        src_indices = src.to(torch.long)
        dst_indices = dst.to(torch.long)

        neg_indices = sample_negatives(src_indices.cpu(), nodes, existing)
        neg_indices = torch.tensor(neg_indices, device=device, dtype=torch.long)

        loss = bpr_loss(emb, (src_indices, dst_indices), (src_indices, neg_indices))
        
        opt.zero_grad()
        loss.backward()
        opt.step()

        epoch_loss += loss.item()

        loop.set_postfix(loss=loss.item())

    epoch_loss /= len(loader)
    print(f'Epoch {epoch + 1}, Loss {epoch_loss:.4f}')
    if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
        save_path = f"lightgcn_epoch_{epoch + 1}.pt"
        torch.save({ 
            'epoch': epoch + 1, 
            'model_state_dict': model.state_dict(), 
            'optimizer_state_dict': opt.state_dict(), 
            'loss': epoch_loss
        }, save_path)
        print(f"Model saved to {save_path}")

model.eval()
with torch.no_grad():
    final_embeddings = model.get_embeddings()
    final_embeddings = final_embeddings.to(device)
    torch.save(final_embeddings, "item_embeddings.pt")
    print("Item embeddings saved.")