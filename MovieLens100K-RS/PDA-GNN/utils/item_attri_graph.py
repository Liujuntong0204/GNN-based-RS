import pandas as pd
import datetime
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops


def extract_year(date_str):
    try:
        return datetime.datetime.strptime(date_str, '%d-%b-%Y').year
    except:
        return None

def get_gragh():
    # 获取数据
    items = pd.read_csv("/kaggle/input/movie-len/items.csv")
    item_count = items['itemId'].max() + 1

    # 获取title边
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    title_embeddings = model.encode(items['Title'].tolist(), convert_to_tensor=False)
    # 计算余弦相似， > 0.7连边
    edges_title = []
    for i in range(item_count):
        for j in range(i + 1, item_count):
            sim = util.cos_sim(title_embeddings[i], title_embeddings[j])[0][0]
            if sim > 0.7:
                edges_title += [(i,j), (j,i)]
    edges_title = pd.DataFrame(edges_title, columns=['source', 'target'])
    edges_title = edges_title.drop_duplicates()
    print("edges_title :", len(edges_title))


    # 获取年份边
    items['Year'] = items['Release Date'].apply(extract_year)
    edges_year = []
    for i in range(item_count):
        year_i = items.iloc[i]['Year']
        if year_i is None:
            continue
        for j in range(i + 1, item_count):
            year_j = items.iloc[j]['Year']
            if year_j is None:
                continue 
            if abs(year_i - year_j) <= 1:
                edges_year += [(i,j), (j,i)]

    edges_year = pd.DataFrame(edges_year, columns=['source', 'target'])
    edges_year = edges_year.drop_duplicates()
    print("edges_year :", len(edges_year))



    # 获取类别边
    categories = ['unknown', 'Action', 'Adventure', 'Animation', "Children's",
                'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                'Film-Noir', 'Horror', 'Musical', 'Mystery',
                'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    edges_categories = []
    for i in range(item_count):
        cats_i = set([cat for cat, val in zip(categories, items.iloc[i, 4:]) if val == 1])
        for j in range(i + 1, item_count):
            cats_j = set([cat for cat, val in zip(categories, items.iloc[j, 4:]) if val == 1])

            if len(cats_i & cats_j) > 0:
                edges_categories += [(i,j), (j,i)]

    edges_categories = pd.DataFrame(edges_categories, columns=['source', 'target'])
    edges_categories = edges_categories.drop_duplicates()
    print("edges_categories :", len(edges_categories))

    return item_count, edges_title, edges_year, edges_categories

def create_data_from_edges(edges_df, num_nodes):
    edge_index = torch.tensor(edges_df[['source', 'target']].values, dtype=torch.long).t().contiguous()
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes) 
    return Data(edge_index=edge_index, num_nodes=num_nodes)