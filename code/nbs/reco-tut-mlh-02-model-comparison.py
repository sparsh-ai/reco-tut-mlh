#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
project_name = "reco-tut-mlh"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)


# In[2]:


if not os.path.exists(project_path):
    get_ipython().system(u'cp /content/drive/MyDrive/mykeys.py /content')
    import mykeys
    get_ipython().system(u'rm /content/mykeys.py')
    path = "/content/" + project_name; 
    get_ipython().system(u'mkdir "{path}"')
    get_ipython().magic(u'cd "{path}"')
    import sys; sys.path.append(path)
    get_ipython().system(u'git config --global user.email "recotut@recohut.com"')
    get_ipython().system(u'git config --global user.name  "reco-tut"')
    get_ipython().system(u'git init')
    get_ipython().system(u'git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git')
    get_ipython().system(u'git pull origin "{branch}"')
    get_ipython().system(u'git checkout main')
else:
    get_ipython().magic(u'cd "{project_path}"')


# In[34]:


get_ipython().system(u'git status')


# In[35]:


get_ipython().system(u'git add . && git commit -m \'commit\' && git push origin "{branch}"')


# In[7]:


import sys
sys.path.insert(0, './code')


# ---

# # Collaborative Filtering Comparison
# 
# In this notebook we compare different recommendation systems starting with the state-of-the-art LightGCN and going back to the winning algorithm for 2009's Netflix Prize competition, SVD++.
# 
# Models include in order are LightGCN, NGCF, SVAE, SVD++, and SVD. Each model has their own individual notebooks where we go more indepth, especially LightGCN and NGCF, where we implemented them from scratch in Tensorflow. 
# 
# The last cell compares the performance of the different models using ranking metrics:
# 
# 
# *   Precision@k
# *   Recall@k
# *   Mean Average Precision (MAP)
# *   Normalized Discounted Cumulative Gain (NDCG)
# 
# where $k=10$
# 
# 

# # Imports

# In[4]:


get_ipython().system(u'pip install -q surprise')


# In[8]:


import math
import numpy as np
import os
import pandas as pd
import random
import requests
import scipy.sparse as sp
import surprise
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.python.framework.ops import disable_eager_execution
from tqdm import tqdm

from utils import stratified_split, numpy_stratified_split
import build_features
import metrics
from models import SVAE
from models.GCN import LightGCN, NGCF


# # Prepare data

# In[9]:


fp = os.path.join('./data/bronze', 'u.data')
raw_data = pd.read_csv(fp, sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'])
print(f'Shape: {raw_data.shape}')
raw_data.sample(10, random_state=123)


# In[10]:


# Load movie titles.
fp = os.path.join('./data/bronze', 'u.item')
movie_titles = pd.read_csv(fp, sep='|', names=['movieId', 'title'], usecols = range(2), encoding='iso-8859-1')
print(f'Shape: {movie_titles.shape}')
movie_titles.sample(10, random_state=123)


# In[15]:


train_size = 0.75
train, test = stratified_split(raw_data, 'userId', train_size)

print(f'Train Shape: {train.shape}')
print(f'Test Shape: {test.shape}')
print(f'Do they have the same users?: {set(train.userId) == set(test.userId)}')


# In[16]:


combined = train.append(test)

n_users = combined['userId'].nunique()
print('Number of users:', n_users)

n_movies = combined['movieId'].nunique()
print('Number of movies:', n_movies)


# In[17]:


# Create DataFrame with reset index of 0-n_movies.
movie_new = combined[['movieId']].drop_duplicates()
movie_new['movieId_new'] = np.arange(len(movie_new))

train_reindex = pd.merge(train, movie_new, on='movieId', how='left')
# Reset index to 0-n_users.
train_reindex['userId_new'] = train_reindex['userId'] - 1  
train_reindex = train_reindex[['userId_new', 'movieId_new', 'rating']]

test_reindex = pd.merge(test, movie_new, on='movieId', how='left')
# Reset index to 0-n_users.
test_reindex['userId_new'] = test_reindex['userId'] - 1
test_reindex = test_reindex[['userId_new', 'movieId_new', 'rating']]

# Create dictionaries so we can convert to and from indexes
item2id = dict(zip(movie_new['movieId'], movie_new['movieId_new']))
id2item = dict(zip(movie_new['movieId_new'], movie_new['movieId']))
user2id = dict(zip(train['userId'], train_reindex['userId_new']))
id2user = dict(zip(train_reindex['userId_new'], train['userId']))


# In[18]:


# Create user-item graph (sparse matix where users are rows and movies are columns.
# 1 if a user reviewed that movie, 0 if they didn't).
R = sp.dok_matrix((n_users, n_movies), dtype=np.float32)
R[train_reindex['userId_new'], train_reindex['movieId_new']] = 1

# Create the adjaceny matrix with the user-item graph.
adj_mat = sp.dok_matrix((n_users + n_movies, n_users + n_movies), dtype=np.float32)

# List of lists.
adj_mat.tolil()
R = R.tolil()

# Put together adjacency matrix. Movies and users are nodes/vertices.
# 1 if the movie and user are connected.
adj_mat[:n_users, n_users:] = R
adj_mat[n_users:, :n_users] = R.T

adj_mat


# In[19]:


# Calculate degree matrix D (for every row count the number of nonzero entries)
D_values = np.array(adj_mat.sum(1))

# Square root and inverse.
D_inv_values = np.power(D_values  + 1e-9, -0.5).flatten()
D_inv_values[np.isinf(D_inv_values)] = 0.0

 # Create sparse matrix with the values of D^(-0.5) are the diagonals.
D_inv_sq_root = sp.diags(D_inv_values)

# Eval (D^-0.5 * A * D^-0.5).
norm_adj_mat = D_inv_sq_root.dot(adj_mat).dot(D_inv_sq_root)


# In[20]:


# to COOrdinate format first ((row, column), data)
coo = norm_adj_mat.tocoo().astype(np.float32)

# create an index that will tell SparseTensor where the non-zero points are
indices = np.mat([coo.row, coo.col]).transpose()

# covert to sparse tensor
A_tilde = tf.SparseTensor(indices, coo.data, coo.shape)
A_tilde


# # Train models

# ## Graph Convoultional Networks (GCNs)

# ### Light Graph Convolution Network (LightGCN)

# In[21]:


light_model = LightGCN(A_tilde,
                 n_users = n_users,
                 n_items = n_movies,
                 n_layers = 3)


# In[22]:


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
light_model.fit(epochs=25, batch_size=1024, optimizer=optimizer)


# ### Neural Graph Collaborative Filtering (NGCF)

# In[23]:


ngcf_model = NGCF(A_tilde,
                  n_users = n_users,
                  n_items = n_movies,
                  n_layers = 3
                  )

ngcf_model.fit(epochs=25, batch_size=1024, optimizer=optimizer)


# ### Recommend with LightGCN and NGCF

# In[24]:


# Convert test user ids to the new ids
users = np.array([user2id[x] for x in test['userId'].unique()])

recs = []
for model in [light_model, ngcf_model]:
    recommendations = model.recommend(users, k=10)
    recommendations = recommendations.replace({'userId': id2user, 'movieId': id2item})
    recommendations = recommendations.merge(movie_titles,
                                                    how='left',
                                                    on='movieId'
                                                    )[['userId', 'movieId', 'title', 'prediction']]

    # Create column with the predicted movie's rank for each user 
    top_k = recommendations.copy()
    top_k['rank'] = recommendations.groupby('userId', sort=False).cumcount() + 1  # For each user, only include movies recommendations that are also in the test set

    recs.append(top_k)


# ## Standard Variational Autoencoder (SVAE)

# In[26]:


# Binarize the data (only keep ratings >= 4)
df_preferred = raw_data[raw_data['rating'] > 3.5]
df_low_rating = raw_data[raw_data['rating'] <= 3.5]

df = df_preferred.groupby('userId').filter(lambda x: len(x) >= 5)
df = df.groupby('movieId').filter(lambda x: len(x) >= 1)

# Obtain both usercount and itemcount after filtering
usercount = df[['userId']].groupby('userId', as_index = False).size()
itemcount = df[['movieId']].groupby('movieId', as_index = False).size()

unique_users =sorted(df.userId.unique())
np.random.seed(123)
unique_users = np.random.permutation(unique_users)

HELDOUT_USERS = 200

# Create train/validation/test users
n_users = len(unique_users)
train_users = unique_users[:(n_users - HELDOUT_USERS * 2)]
val_users = unique_users[(n_users - HELDOUT_USERS * 2) : (n_users - HELDOUT_USERS)]
test_users = unique_users[(n_users - HELDOUT_USERS):]

train_set = df.loc[df['userId'].isin(train_users)]
val_set = df.loc[df['userId'].isin(val_users)]
test_set = df.loc[df['userId'].isin(test_users)]
unique_train_items = pd.unique(train_set['movieId'])
val_set = val_set.loc[val_set['movieId'].isin(unique_train_items)]
test_set = test_set.loc[test_set['movieId'].isin(unique_train_items)]

# Instantiate the sparse matrix generation for train, validation and test sets
# use list of unique items from training set for all sets
am_train = build_features.AffinityMatrix(df=train_set, items_list=unique_train_items)
am_val = build_features.AffinityMatrix(df=val_set, items_list=unique_train_items)
am_test = build_features.AffinityMatrix(df=test_set, items_list=unique_train_items)

# Obtain the sparse matrix for train, validation and test sets
train_data, _, _ = am_train.gen_affinity_matrix()
val_data, val_map_users, val_map_items = am_val.gen_affinity_matrix()
test_data, test_map_users, test_map_items = am_test.gen_affinity_matrix()

# Split validation and test data into training and testing parts
val_data_tr, val_data_te = numpy_stratified_split(val_data, ratio=0.75, seed=123)
test_data_tr, test_data_te = numpy_stratified_split(test_data, ratio=0.75, seed=123)

# Binarize train, validation and test data
train_data = np.where(train_data > 3.5, 1.0, 0.0)
val_data = np.where(val_data > 3.5, 1.0, 0.0)
test_data = np.where(test_data > 3.5, 1.0, 0.0)

# Binarize validation data
val_data_tr = np.where(val_data_tr > 3.5, 1.0, 0.0)
val_data_te_ratings = val_data_te.copy()
val_data_te = np.where(val_data_te > 3.5, 1.0, 0.0)

# Binarize test data: training part 
test_data_tr = np.where(test_data_tr > 3.5, 1.0, 0.0)

# Binarize test data: testing part (save non-binary version in the separate object, will be used for calculating NDCG)
test_data_te_ratings = test_data_te.copy()
test_data_te = np.where(test_data_te > 3.5, 1.0, 0.0)

# retrieve real ratings from initial dataset 
test_data_te_ratings=pd.DataFrame(test_data_te_ratings)
val_data_te_ratings=pd.DataFrame(val_data_te_ratings)

for index,i in df_low_rating.iterrows():
    user_old= i['userId'] # old value 
    item_old=i['movieId'] # old value 

    if (test_map_users.get(user_old) is not None)  and (test_map_items.get(item_old) is not None) :
        user_new=test_map_users.get(user_old) # new value 
        item_new=test_map_items.get(item_old) # new value 
        rating=i['rating'] 
        test_data_te_ratings.at[user_new,item_new]= rating   

    if (val_map_users.get(user_old) is not None)  and (val_map_items.get(item_old) is not None) :
        user_new=val_map_users.get(user_old) # new value 
        item_new=val_map_items.get(item_old) # new value 
        rating=i['rating'] 
        val_data_te_ratings.at[user_new,item_new]= rating   


val_data_te_ratings=val_data_te_ratings.to_numpy()    
test_data_te_ratings=test_data_te_ratings.to_numpy()    


# In[27]:


disable_eager_execution()
svae_model = SVAE.StandardVAE(n_users=train_data.shape[0],
                                   original_dim=train_data.shape[1], 
                                   intermediate_dim=200, 
                                   latent_dim=64, 
                                   n_epochs=400, 
                                   batch_size=100, 
                                   k=10,
                                   verbose=0,
                                   seed=123,
                                   drop_encoder=0.5,
                                   drop_decoder=0.5,
                                   annealing=False,
                                   beta=1.0
                                   )

svae_model.fit(x_train=train_data,
          x_valid=val_data,
          x_val_tr=val_data_tr,
          x_val_te=val_data_te_ratings,
          mapper=am_val
          )


# ### Recommend with SVAE

# In[28]:


# Model prediction on the training part of test set 
top_k =  svae_model.recommend_k_items(x=test_data_tr,k=10,remove_seen=True)

# Convert sparse matrix back to df
recommendations = am_test.map_back_sparse(top_k, kind='prediction')
test_df = am_test.map_back_sparse(test_data_te_ratings, kind='ratings') # use test_data_te_, with the original ratings

# Create column with the predicted movie's rank for each user 
top_k = recommendations.copy()
top_k['rank'] = recommendations.groupby('userId', sort=False).cumcount() + 1  # For each user, only include movies recommendations that are also in the test set

recs.append(top_k)


# ## Singular Value Decomposition (SVD)

# ### SVD++

# In[29]:


surprise_train = surprise.Dataset.load_from_df(train.drop('timestamp', axis=1), reader=surprise.Reader('ml-100k')).build_full_trainset()
svdpp = surprise.SVDpp(random_state=0, n_factors=64, n_epochs=10, verbose=True)
svdpp.fit(surprise_train)


# ### SVD

# In[30]:


svd = surprise.SVD(random_state=0, n_factors=64, n_epochs=10, verbose=True)
svd.fit(surprise_train)


# ### Recommend with SVD++ and SVD

# In[31]:


for model in [svdpp, svd]:
    predictions = []
    users = train['userId'].unique()
    items = train['movieId'].unique()

    for user in users:
            for item in items:
                predictions.append([user, item, model.predict(user, item).est])

    predictions = pd.DataFrame(predictions, columns=['userId', 'movieId', 'prediction'])

    # Remove movies already seen by users
    # Create column of all 1s
    temp = train[['userId', 'movieId']].copy()
    temp['seen'] = 1

    # Outer join and remove movies that have alread been seen (seen=1)
    merged = pd.merge(temp, predictions, on=['userId', 'movieId'], how="outer")
    merged = merged[merged['seen'].isnull()].drop('seen', axis=1)

    # Create filter for users that appear in both the train and test set
    common_users = set(test['userId']).intersection(set(predictions['userId']))

    # Filter the test and predictions so they have the same users between them
    test_common = test[test['userId'].isin(common_users)]
    svd_pred_common = merged[merged['userId'].isin(common_users)]

    if len(set(merged['userId'])) != len(set(test['userId'])):
        print('Number of users in train and test are NOT equal')
        print(f"# of users in train and test respectively: {len(set(merged['userId']))}, {len(set(test['userId']))}")
        print(f"# of users in BOTH train and test: {len(set(svd_pred_common['userId']))}")
        continue
        
    # From the predictions, we want only the top k for each user,
    # not all the recommendations.
    # Extract the top k recommendations from the predictions
    top_movies = svd_pred_common.groupby('userId', as_index=False).apply(lambda x: x.nlargest(10, 'prediction')).reset_index(drop=True)
    top_movies['rank'] = top_movies.groupby('userId', sort=False).cumcount() + 1
    
    top_k = top_movies.copy()
    top_k['rank'] = top_movies.groupby('userId', sort=False).cumcount() + 1  # For each user, only include movies recommendations that are also in the test set
    
    recs.append(top_k)


# # Compare performance

# Looking at all 5 of our models, we can see that the state-of-the-art model LightGCN vastly outperforms all other models. When compared to SVD++, a widely used algorithm during the Netflix Prize competition, LightGCN achieves an increase in **Percision@k by 29%, Recall@k by 18%, MAP by 12%, and NDCG by 35%**.
# 
# NGCF is the older sister model to LightGCN, but only by a single year. We can see how LightGCN improves in ranking metrics compared to NGCF by simply removing unnecessary operations. 
# 
# In conclusion, this demonstrates how far recommendation systems have advanced since 2009, and how new model architectures with notable performance increases can be developed in the span of just 1-2 years.

# In[32]:


model_names = ['LightGCN', 'NGCF', 'SVAE', 'SVD++', 'SVD']
comparison = pd.DataFrame(columns=['Algorithm', 'Precision@k', 'Recall@k', 'MAP', 'NDCG'])

# Convert test user ids to the new ids
users = np.array([user2id[x] for x in test['userId'].unique()])

for rec, name in zip(recs, model_names):
    tester = test_df if name == 'SVAE' else test

    pak = metrics.precision_at_k(rec, tester, 'userId', 'movieId', 'rank')
    rak = metrics.recall_at_k(rec, tester, 'userId', 'movieId', 'rank')
    map = metrics.mean_average_precision(rec, tester, 'userId', 'movieId', 'rank')
    ndcg = metrics.ndcg(rec, tester, 'userId', 'movieId', 'rank')

    comparison.loc[len(comparison)] = [name, pak, rak, map, ndcg]


# In[33]:


comparison


# # References:
# 
# 1.   Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang & Meng Wang, LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation, 2020, https://arxiv.org/abs/2002.02126
# 2.   Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, & Tata-Seng Chua, Neural Graph Collaorative Filtering, 2019, https://arxiv.org/abs/1905.08108
# 3.   Microsoft SVAE implementation: https://github.com/microsoft/recommenders/blob/main/examples/02_model_collaborative_filtering/standard_vae_deep_dive.ipynb
# 4. Simon Gower, Netflix Prize and SVD, 2014, https://www.semanticscholar.org/paper/Netflix-Prize-and-SVD-Gower/ce7b81b46939d7852dbb30538a7796e69fdd407c
# 
