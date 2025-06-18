import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
from utils.item_attri_graph import get_gragh, create_data_from_edges
from utils.sampler import sample_negatives
from Model.PDA_GNN import PDA_GNN, bpr_loss

item_count, edges_title, edges_year, edges_categories = get_gragh()
data_title = create_data_from_edges(edges_title, item_count)
data_year = create_data_from_edges(edges_year, item_count)
data_cat = create_data_from_edges(edges_categories, item_count)

# 参数
embedding_dim = 64
num_layers = 2
epochs = 30
batch_size = 10
num_neighbors = [10,10]
learning_rate = 0.001

loader_title = NeighborLoader(
    data=data_title,
    num_neighbors=[5, 5],
    batch_size=batch_size,
    shuffle=True,
    input_nodes=None
)
print("loader_title")
loader_year = NeighborLoader(
    data=data_year,
    num_neighbors=num_neighbors,
    batch_size=batch_size,
    shuffle=True,
    input_nodes=None
)
print("loader_year")
loader_cat = NeighborLoader(
    data=data_cat,
    num_neighbors=num_neighbors,
    batch_size=batch_size,
    shuffle=True,
    input_nodes=None
)
print("loader_cat")


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = PDA_GNN(num_nodes=item_count, embedding_dim=embedding_dim, num_layers=num_layers).to(device)
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

existing_title = set([ (s.item(), d.item()) for s,d in data_title.edge_index.t() ])
existing_year = set([ (s.item(), d.item()) for s,d in data_year.edge_index.t() ])
existing_cat = set([ (s.item(), d.item()) for s,d in data_cat.edge_index.t() ])
nodes = list(range(item_count))

model.train()
for epoch in range(epochs):
    epoch_loss = 0
    loop = tqdm(zip(loader_title, loader_year, loader_cat), total=len(loader_title), desc=f"Epoch {epoch+1}/{epochs}")
    for batch_title, batch_year, batch_cat in loop:
        batch_title = batch_title.to(device)
        batch_year = batch_year.to(device)
        batch_cat = batch_cat.to(device)

        emb = model(batch_title.edge_index, batch_year.edge_index, batch_cat.edge_index)
        emb = F.normalize(emb, dim=1)

        # Title图正负采样
        src_t, dst_t = batch_title.edge_index
        neg_dst_t = sample_negatives(src_t.cpu(), nodes, existing_title)
        src_t = src_t.to(device=device, dtype=torch.long)
        dst_t = dst_t.to(device=device, dtype=torch.long)
        neg_dst_t = torch.tensor(neg_dst_t, device=device, dtype=torch.long)

        loss_title = bpr_loss(emb, (src_t, dst_t), (src_t, neg_dst_t))

        # Year图正负采样
        src_y, dst_y = batch_year.edge_index
        neg_dst_y = sample_negatives(src_y.cpu(), nodes, existing_year)
        src_y = src_y.to(device=device, dtype=torch.long)
        dst_y = dst_y.to(device=device, dtype=torch.long)
        neg_dst_y = torch.tensor(neg_dst_y, device=device, dtype=torch.long)

        loss_year = bpr_loss(emb, (src_y, dst_y), (src_y, neg_dst_y))

        # Category图正负采样
        src_c, dst_c = batch_cat.edge_index
        neg_dst_c = sample_negatives(src_c.cpu(), nodes, existing_cat)
        src_c = src_c.to(device=device, dtype=torch.long)
        dst_c = dst_c.to(device=device, dtype=torch.long)
        neg_dst_c = torch.tensor(neg_dst_c, device=device, dtype=torch.long)

        loss_cat = bpr_loss(emb, (src_c, dst_c), (src_c, neg_dst_c))

        weights = F.softmax(model.alpha, dim=0)
        weights = torch.clamp(weights, min=1e-4)  
        weights = weights / weights.sum() 
        loss = weights[0]*loss_title + weights[1]*loss_year + weights[2]*loss_cat

        opt.zero_grad()
        loss.backward()
        opt.step()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    epoch_loss /= len(loader_title)
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

# 保存属性嵌入
model.eval()
with torch.no_grad():
    full_emb = model.get_final_embeddings(
        data_title.edge_index.to(device),
        data_year.edge_index.to(device),
        data_cat.edge_index.to(device))
    torch.save(full_emb.cpu(), "attribute_embeddings.pt")
    print("属性嵌入保存完毕")