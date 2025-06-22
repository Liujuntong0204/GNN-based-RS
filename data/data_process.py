import pandas as pd

#  users.dat
user_df = pd.read_csv(
    "/kaggle/input/movielens-1m-dataset/users.dat",
    sep="::",
    engine="python",
    names=["UserID", "Gender", "Age", "Occupation", "Zip-code"],
    header=None
)
print(len(user_df))

# 唯一用户ID
raw_user_ids = user_df["UserID"].unique()
print(len(raw_user_ids))

# 构建索引
user_id2idx = {raw_id: idx for idx, raw_id in enumerate(raw_user_ids)}
# print(user_id2idx)
idx2user_id = {idx: raw_id for raw_id, idx in user_id2idx.items()}
# print(idx2user_id)

import pandas as pd

# movies.dat
movie_df = pd.read_csv(
    "/kaggle/input/movielens-1m-dataset/movies.dat",
    sep="::",
    engine="python",
    names=["MovieID", "Title", "Genres"],
    header=None,
    encoding='ISO-8859-1'
)
print(len(movie_df))

# 唯一电影ID
raw_movie_ids = movie_df["MovieID"].unique()
print(len(raw_movie_ids))

# 构建映射
movie_id2idx = {raw_id: idx for idx, raw_id in enumerate(raw_movie_ids)}
# print(movie_id2idx)
idx2movie_id = {idx: raw_id for raw_id, idx in movie_id2idx.items()}
# print(idx2movie_id)

# 保存映射
import pickle

mapping_dict = {
    "user_id2idx": user_id2idx,
    "idx2user_id": idx2user_id,
    "movie_id2idx": movie_id2idx,
    "idx2movie_id": idx2movie_id
}

with open("id_mappings.pkl", "wb") as f:
    pickle.dump(mapping_dict, f)

# # 加载使用
# import pickle
# mapping_dict = {}
# with open("/kaggle/input/id-idx-mapping/id_mappings.pkl", "rb") as f:
#     mapping_dict = pickle.load(f)
# user_id2idx = mapping_dict["user_id2idx"]
# idx2user_id = mapping_dict["idx2user_id"]
# movie_id2idx = mapping_dict["movie_id2idx"]
# idx2movie_id = mapping_dict["idx2movie_id"]



import pandas as pd
import re

import pickle
mapping_dict = {}
with open("/kaggle/input/id-idx-mapping/id_mappings.pkl", "rb") as f:
    mapping_dict = pickle.load(f)
user_id2idx = mapping_dict["user_id2idx"]
idx2user_id = mapping_dict["idx2user_id"]
movie_id2idx = mapping_dict["movie_id2idx"]
idx2movie_id = mapping_dict["idx2movie_id"]


ratings_file = "/kaggle/input/movielens-1m-dataset/ratings.dat"
item_file = "/kaggle/input/movielens-1m-dataset/movies.dat"

# 评分数据 ::分隔
ratings = pd.read_csv(ratings_file, sep='::', engine='python', header=None,
                      names=['userId', 'itemId', 'score', 'time'])
print("rate_count: ", len(ratings))
ratings['userIdx'] = ratings['userId'].map(user_id2idx)
print("userIdx - userId:")
print(ratings[['userIdx', 'userId']])
ratings['itemIdx'] = ratings['itemId'].map(movie_id2idx)
print("itemIdx - itemId")
print(ratings[['itemIdx', 'itemId']])


# 读取电影数据
item_columns = ['itemId', 'Title', 'Genres']
items = pd.read_csv(item_file, sep='::', engine='python', header=None, names=item_columns, encoding='ISO-8859-1')
items['itemIdx'] = items['itemId'].map(movie_id2idx)

# 获取年份
def extract_year(title):
    match = re.search(r'\((\d{4})\)', title)
    return match.group(1) if match else None

# 移除年份
def remove_year_from_title(title):
    return re.sub(r'\s*\(\d{4}\)', '', title)

# 添加 Year 列
items['Year'] = items['Title'].apply(extract_year)
items['Title'] = items['Title'].apply(remove_year_from_title)


all_genres = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

for genre in all_genres:
    items[genre] = 0

# 对应 genre 设为 1
def set_genres(row):
    genres = row['Genres'].split('|')
    for genre in genres:
        if genre in all_genres:
            row[genre] = 1
    return row

items = items.apply(set_genres, axis=1)
items = items[['itemId', 'itemIdx', 'Title', 'Year'] + all_genres]

print("item columns:", items.columns.tolist())
print("item_count: ", len(items))
print("itemIdx - itemId: ")
print(items[['itemIdx', 'itemId']])

# 划分训练集和测试集
train_list = []
val_list = []

for user_id, group in ratings.groupby('userId'):
    group_sorted = group.sort_values(by='time')  # 按时间排序
    split_idx = int(len(group_sorted) * 0.8)
    train_list.append(group_sorted.iloc[:split_idx])
    val_list.append(group_sorted.iloc[split_idx:])

train_ratings = pd.concat(train_list, ignore_index=True)
val_ratings = pd.concat(val_list, ignore_index=True)

print("rating train columns:", train_ratings.columns.tolist())
print("train rating num: ", len(train_ratings))
print("rating test columns:", val_ratings.columns.tolist())
print("test rating num: ", len(val_ratings))

train_ratings.to_csv("train.csv", index=False)
val_ratings.to_csv("validation.csv", index=False)
items.to_csv("items.csv", index=False)