# 获取图
import itertools
import pandas as pd
def get_action_graph():
    train_df = pd.read_csv("/kaggle/input/movie-len/train.csv")

    item_indices = train_df.groupby('userId')['itemId'].apply(lambda x: list(x)).reset_index()

    items_df = pd.read_csv("/kaggle/input/movie-len/items.csv")
    item_count = items_df['itemId'].max()+1
    print("itemcount:", item_count)

    edges = []
    for row in item_indices.itertuples():
        items = row[2]  
        for i, j in itertools.combinations(items, 2):
            edges.append((i, j))
            edges.append((j, i))  

    edges = pd.DataFrame(edges, columns=['source', 'target'])

    edges = edges.drop_duplicates()
    print("Number of edges :", len(edges))

    return item_count, edges

