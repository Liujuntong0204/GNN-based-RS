# ITEM-BASED-RS —— 基于物品视角建模的推荐方法

本项目实现了一种结合物品行为信息与属性信息的推荐方法，并进行了初步实验验证。该方法的核心思想是从物品视角出发，解耦学习行为嵌入与属性嵌入，并通过注意力机制完成融合。



## 方法简介

本方法从物品视角出发，构建融合多源信息的推荐模型，主要包括以下几个步骤：

**1. 行为嵌入建模（Behavior Embedding）**
 基于用户-物品交互数据构建物品共现图（item-item graph），利用图神经网络对物品在该图上的行为表示进行学习。

**2. 属性嵌入建模（Attribute Embedding）**
 针对物品的多个属性（如标题、年份、类别），分别构建属性图，各属性解耦传播，最终通过注意力机制将多种属性嵌入融合得到物品的属性表示。

**3. 行为与属性融合（Embedding Fusion）**
 将上述两种嵌入表示通过注意力机制进行加权融合，生成物品的最终嵌入。

**4. 用户表示建模（User Embedding）**
 使用用户注意力池化机制，聚合用户历史交互过的物品的最终嵌入，得到用户嵌入表示。

**5. 推荐生成（Recommendation）**
 利用用户与物品嵌入之间的点积相似度进行推荐。



## 项目结构

```
ITEM-BASED-RS/
├── data/                    # 数据集与数据处理
├── evaluation/              # 推荐评估指标计算
├── item_act_emb/            # 行为嵌入模块
├── item_attri_emb/          # 属性嵌入模块
├── item_final_emb/          # 行为与属性融合模块
├── user_final_emb/          # 用户嵌入模块
├── saved/                   # 存储中间嵌入结果
├── README
└── .gitignore
```



## 初步实验结果

| 指标名称         | 值     |
| ---------------- | ------ |
| **Recall@10**    | 0.0471 |
| **Precision@10** | 0.1151 |
| **NDCG@10**      | 0.1241 |
| **HitRate@10**   | 0.5185 |

> 当前结果仍有提升空间，仅作为方法验证和预研参考，后续将尝试改进属性表示、融合策略和图结构建模以提升性能。



## 项目说明

本仓库为推荐系统的预研性探索工作，主要用于验证基于物品视角的建模方案可行性。当前模型和结构为初始版本，后续将考虑引入更多高阶建模机制，以进一步提升性能与实用性。



本项目使用的数据集为  [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)，由 GroupLens Research 提供。若在论文或出版物中使用该数据集，请引用以下文献：

> F. Maxwell Harper and Joseph A. Konstan.
>  *The MovieLens Datasets: History and Context.*
>  ACM Transactions on Interactive Intelligent Systems (TiiS), 5(4), Article 19 (2015).
>  https://doi.org/10.1145/2827872

