import pandas as pd
import os

# 文件路径
train_file = "/kaggle/input/movielens-100k-dataset/ml-100k/ua.base"
valitation_file = "/kaggle/input/movielens-100k-dataset/ml-100k/ua.test"
item_file = "/kaggle/input/movielens-100k-dataset/ml-100k/u.item"

# 读取数据 \t分隔
train_ratings = pd.read_csv(train_file, sep='\t', header=None, names=['userId', 'itemId', 'score', 'time'])
val_ratings = pd.read_csv(valitation_file, sep='\t', header=None, names=['userId', 'itemId', 'score', 'time'])

# 读取电影数据  |分隔
item_columns = ['itemId', 'Title', 'Release Date', 'Video Release Date', 'IMDb URL'] + \
               ['unknown', 'Action', 'Adventure', 'Animation', "Children's",
                'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                'Film-Noir', 'Horror', 'Musical', 'Mystery',
                'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv(item_file, sep='|', header=None, names=item_columns, encoding='ISO-8859-1')
items = items.copy()

# 索引从0开始
train_ratings['userId'] -= 1
train_ratings['itemId'] -= 1
val_ratings['userId'] -= 1
val_ratings['itemId'] -= 1
items['itemId'] -= 1



items = items[['itemId', 'Title', 'Release Date', 
               'unknown', 'Action', 'Adventure', 'Animation', "Children's",
               'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
               'Film-Noir', 'Horror', 'Musical', 'Mystery',
               'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']]

print("item:  ", items.columns)


train_ratings.to_csv("train.csv", index=False)
val_ratings.to_csv("validation.csv", index=False)
items.to_csv("items.csv", index=False)