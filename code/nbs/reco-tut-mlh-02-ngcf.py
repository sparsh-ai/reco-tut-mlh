#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
project_name = "reco-tut-mlh"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)


# In[3]:


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


# In[37]:


get_ipython().system(u'git status')


# In[38]:


get_ipython().system(u'git add . && git commit -m \'commit\' && git push origin "{branch}"')


# In[7]:


import sys
sys.path.insert(0,'./code')


# ---

# # Neural Graph Collaborative Filtering (NGCF)
# 
# This is a TensorFlow implementation of NGCF with a custom training loop.
# 
# Neural Graph Collaborative Filtering (NGCF) is a state-of-the-art GCN-based recommender model that takes advantage of graph structure and is a precursor to the superior LightGCN. In this notebook, we construct and train an NGCF model and evaluate its performance.

# # Imports

# In[16]:


import math
import numpy as np
import os
import pandas as pd
import random
import requests
import scipy.sparse as sp
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.utils import Progbar
from tqdm import tqdm

import metrics
from utils import stratified_split


# # Prepare data
# 
# This NGCF implementation takes an adjacency matrix in a sparse tensor format as input.
# 
# In preparation of the data for NGCF, we must:
# 
# 
# *   Download the data
# *   Stratified train test split
# *   Create a normalized adjacency matrix
# *   Convert to tensor
# 
# 

# ## Load data
# 
# The data we use is the benchmark MovieLens 100K Dataset, with 100k ratings, 1000 users, and 1700 movies.

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


# ## Train test split
# 
# We split the data using a stratified split so the users in the training set are also the same users in the test set. NGCF is not able to generate recommendations for users not yet seen in the training set.
# 
# Here we will have a training size of 75%

# In[17]:


train_size = 0.75
train, test = stratified_split(raw_data, 'userId', train_size)

print(f'Train Shape: {train.shape}')
print(f'Test Shape: {test.shape}')
print(f'Do they have the same users?: {set(train.userId) == set(test.userId)}')


# ## Reindex
# 
# Reset the index of users and movies from 0-n for both the training and test data. This is to allow better tracking of users and movies. Dictionaries are created so we can easily translate back and forth from the old index to the new index.
# 
# We would also normally remove users with no ratings, but in this case, all entries have a user and a rating between 1-5.
# 
# 

# In[18]:


combined = train.append(test)

n_users = combined['userId'].nunique()
print('Number of users:', n_users)

n_movies = combined['movieId'].nunique()
print('Number of movies:', n_movies)


# In[19]:


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


# In[20]:


# Keep track of which movies each user has reviewed.
# To be used later in training the NGCF.
interacted = (
    train_reindex.groupby("userId_new")["movieId_new"]
    .apply(set)
    .reset_index()
    .rename(columns={"movieId_new": "movie_interacted"})
)


# ## Adjacency matrix
# 
# In our case, nodes are both users and movies. Rows and columns consist of ALL the nodes and for every connection (reviewed movie) there is the value 1.
# 
# To first create the adjacency matrix we first create a user-item graph where similar to the adjacency matrix, connected users and movies are represented as 1 in a sparse array. Unlike the adjacency matrix, a user-item graph only has users for the columns/rows and items as the other, whereas the adjacency matrix has both users and items concatenated as rows and columns.
# 
# 
# In this case, because the graph is undirected (meaning the connections between nodes do not have a specified direction)
# the adjacency matrix is symmetric. We use this to our advantage by transposing the user-item graph to create the adjacency matrix.
# 
# Our adjacency matrix will not include self-connections where each node is connected to itself.

# ### Create adjacency matrix

# In[21]:


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


# ### Normalize adjacency matrix
# 
# This helps numerically stabilize values when repeating graph convolution operations, avoiding the scale of the embeddings increasing or decreasing.
# 
# $\tilde{A} = D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$
# 
# $D$ is the degree/diagonal matrix where it is zero everywhere but its diagonal. The diagonal has the value of the neighborhood size of each node (how many other nodes that node connects to)
# 
# 
# $D^{-\frac{1}{2}}$ on the left side scales $A$ by the source node, while $D^{-\frac{1}{2}}$ right side scales by the neighborhood size of the destination node rather than the source node.
# 
# 
# 

# In[22]:


# Calculate degree matrix D (for every row count the number of nonzero entries)
D_values = np.array(adj_mat.sum(1))

# Square root and inverse.
D_inv_values = np.power(D_values  + 1e-9, -0.5).flatten()
D_inv_values[np.isinf(D_inv_values)] = 0.0

 # Create sparse matrix with the values of D^(-0.5) are the diagonals.
D_inv_sq_root = sp.diags(D_inv_values)

# Eval (D^-0.5 * A * D^-0.5).
norm_adj_mat = D_inv_sq_root.dot(adj_mat).dot(D_inv_sq_root)


# ### Convert to tensor

# In[23]:


# to COOrdinate format first ((row, column), data)
coo = norm_adj_mat.tocoo().astype(np.float32)

# create an index that will tell SparseTensor where the non-zero points are
indices = np.mat([coo.row, coo.col]).transpose()

# covert to sparse tensor
A_tilde = tf.SparseTensor(indices, coo.data, coo.shape)
A_tilde


# # NGCF
# 
# NGCF performs neighbor aggregation while keeping self-connections, feature transformation, and nonlinear activation. This means there is an additional weight matrix at the end of every convolution.
# 
# Neighbor aggregation is done through graph convolutions to learn embeddings that represent nodes. The size of the embeddings can be changed to whatever number. In this notebook, we set the embedding dimension to 64.
# 
# In matrix form, graph convolution can be thought of as matrix multiplication. In the implementation we create a graph convolution layer that performs just this, allowing us to stack as many graph convolutions as we want. We have the number of layers as 3 in this notebook.
# 

# In[24]:


class GraphConv(tf.keras.layers.Layer):
    def __init__(self, adj_mat):
        super(GraphConv, self).__init__()
        self.adj_mat = adj_mat

    def build(self, input_shape):
        self.W = self.add_weight('kernel',
                                      shape=[int(input_shape[-1]),
                                             int(input_shape[-1])])

    def call(self, ego_embeddings):
        pre_embed = tf.sparse.sparse_dense_matmul(self.adj_mat, ego_embeddings)
        return tf.transpose(tf.matmul(self.W, pre_embed, transpose_a=False, transpose_b=True))


# In[25]:


class NGCF(tf.keras.Model):
    def __init__(self, adj_mat, n_users, n_items, n_layers=3, emb_dim=64, decay=0.0001):
        super(NGCF, self).__init__()
        self.adj_mat = adj_mat
        self.R = tf.sparse.to_dense(adj_mat)[:n_users, n_users:]
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.emb_dim = emb_dim
        self.decay = decay

        # Initialize user and item embeddings.
        initializer = tf.keras.initializers.GlorotNormal()
        self.user_embedding = tf.Variable(
            initializer([self.n_users, self.emb_dim]), name='user_embedding'
        )
        self.item_embedding = tf.Variable(
            initializer([self.n_items, self.emb_dim]), name='item_embedding'
        )

        # Stack graph convolutional layers.
        self.gcn = []
        for layer in range(n_layers):
            self.gcn.append(GraphConv(adj_mat)) 
            self.gcn.append(tf.keras.layers.LeakyReLU())

    def call(self, user_emb, item_emb):
        output_embeddings = tf.concat([user_emb, item_emb], axis=0)
        all_embeddings = [output_embeddings]

        # Graph convolutions.
        for i in range(0, self.n_layers):
            output_embeddings = self.gcn[i](output_embeddings)
            all_embeddings += [output_embeddings]

        # Compute the mean of all layers
        all_embeddings = tf.stack(all_embeddings, axis=1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)

        # Split into users and items embeddings
        new_user_embeddings, new_item_embeddings = tf.split(
            all_embeddings, [self.n_users, self.n_items], axis=0
        )

        return new_user_embeddings, new_item_embeddings

    def recommend(self, users, k):
        # Calculate the scores.
        new_user_embed, new_item_embed = model(self.user_embedding, self.item_embedding)
        user_embed = tf.nn.embedding_lookup(new_user_embed, users)
        test_scores = tf.matmul(user_embed, new_item_embed, transpose_a=False, transpose_b=True)
        test_scores = np.array(test_scores)

        # Remove movies already seen.
        test_scores += sp.csr_matrix(self.R)[users, :] * -np.inf

        # Get top movies.
        test_user_idx = np.arange(test_scores.shape[0])[:, None]
        top_items = np.argpartition(test_scores, -k, axis=1)[:, -k:]
        top_scores = test_scores[test_user_idx, top_items]
        sort_ind = np.argsort(-top_scores)
        top_items = top_items[test_user_idx, sort_ind]
        top_scores = top_scores[test_user_idx, sort_ind]
        top_items, top_scores = np.array(top_items), np.array(top_scores)

        # Create Dataframe with recommended movies.
        topk_scores = pd.DataFrame(
            {
                'userId': np.repeat(users, top_items.shape[1]),
                'movieId': top_items.flatten(),
                'prediction': top_scores.flatten(),
            }
        )

        return topk_scores


# ## Custom training
# 
# For training, we batch a number of users from the training set and sample a single positive item (movie that has been reviewed) and a single negative item (movie that has not been reviewed) for each user.

# In[26]:


N_LAYERS = 5
EMBED_DIM = 64
DECAY = 0.0001
EPOCHS = 50
BATCH_SIZE = 1024
LEARNING_RATE = 1e-2

# We expect this # of parameters in our model.
print(f'Parameters: {EMBED_DIM**2 + EMBED_DIM * (n_users + n_movies)}')


# In[27]:


# Initialize model.
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model = NGCF(A_tilde,
                 n_users = n_users,
                 n_items = n_movies,
                 n_layers = N_LAYERS,
                 emb_dim = EMBED_DIM,
                 decay = DECAY)


# In[28]:


get_ipython().run_cell_magic(u'time', u'', u"# Custom training loop from scratch.\nfor epoch in range(1, EPOCHS + 1):\n    print('Epoch %d/%d' % (epoch, EPOCHS))\n    n_batch = train_reindex.shape[0] // BATCH_SIZE + 1\n    bar = Progbar(n_batch, stateful_metrics='training loss')\n    for idx in range(1, n_batch + 1):\n        # Sample batch_size number of users with positive and negative items.\n        indices = range(n_users)\n        if n_users < BATCH_SIZE:\n            users = np.array([random.choice(indices) for _ in range(BATCH_SIZE)])\n        else:\n            users = np.array(random.sample(indices, BATCH_SIZE))\n\n        def sample_neg(x):\n            while True:\n                neg_id = random.randint(0, n_movies - 1)\n                if neg_id not in x:\n                    return neg_id\n\n        # Sample a single movie for each user that the user did and did not review.\n        interact = interacted.iloc[users]\n        pos_items = interact['movie_interacted'].apply(lambda x: random.choice(list(x)))\n        neg_items = interact['movie_interacted'].apply(lambda x: sample_neg(x))\n\n        users, pos_items, neg_items = users, np.array(pos_items), np.array(neg_items)\n\n        with tf.GradientTape() as tape:\n            # Call NGCF with user and item embeddings.\n            new_user_embeddings, new_item_embeddings = model(\n                model.user_embedding, model.item_embedding\n            )\n\n            # Embeddings after convolutions.\n            user_embeddings = tf.nn.embedding_lookup(new_user_embeddings, users)\n            pos_item_embeddings = tf.nn.embedding_lookup(new_item_embeddings, pos_items)\n            neg_item_embeddings = tf.nn.embedding_lookup(new_item_embeddings, neg_items)\n\n            # Initial embeddings before convolutions.\n            old_user_embeddings = tf.nn.embedding_lookup(\n                model.user_embedding, users\n            )\n            old_pos_item_embeddings = tf.nn.embedding_lookup(\n                model.item_embedding, pos_items\n            )\n            old_neg_item_embeddings = tf.nn.embedding_lookup(\n                model.item_embedding, neg_items\n            )\n\n            # Calculate loss.\n            pos_scores = tf.reduce_sum(\n                tf.multiply(user_embeddings, pos_item_embeddings), axis=1\n            )\n            neg_scores = tf.reduce_sum(\n                tf.multiply(user_embeddings, neg_item_embeddings), axis=1\n            )\n            regularizer = (\n                tf.nn.l2_loss(old_user_embeddings)\n                + tf.nn.l2_loss(old_pos_item_embeddings)\n                + tf.nn.l2_loss(old_neg_item_embeddings)\n            )\n            regularizer = regularizer / BATCH_SIZE\n            mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))\n            emb_loss = DECAY * regularizer\n            loss = mf_loss + emb_loss\n\n        # Retreive and apply gradients.\n        grads = tape.gradient(loss, model.trainable_weights)\n        optimizer.apply_gradients(zip(grads, model.trainable_weights))\n\n        bar.add(1, values=[('training loss', float(loss))])")


# # Recommend

# In[29]:


# Convert test user ids to the new ids
users = np.array([user2id[x] for x in test['userId'].unique()])

recommendations = model.recommend(users, k=10)
recommendations = recommendations.replace({'userId': id2user, 'movieId': id2item})
recommendations = recommendations.merge(
    movie_titles, how='left', on='movieId'
)[['userId', 'movieId', 'title', 'prediction']]
recommendations.head(15)


# # Evaluation Metrics
# 
# The performance of our model is evaluated using the test set, which consists of the exact same users in the training set but with movies the users have reviewed that the model has not seen before.
# 
# A good model will recommend movies that the user has also reviewed in the test set.

# ## Precision@k
# 
# Out of the movies that are recommended, what proportion is relevant. Relevant in this case is if the user has reviewed the movie.
# 
# A precision@10 of about 0.35 means that about 35% of the recommendations from NGCF are relevant to the user. In other words, out of the 10 recommendations made, on average a user will have 4 movies that are actually relevant.
# 
# ## Recall@k
# 
# Out of all the relevant movies (in the test set), how many are recommended.
# 
# A recall@10 of about 0.19 means that about 19% of the relevant movies were recommended by NGCF. By definition you can see how even if all the recommendations made were relevant, recall@k is capped by k. A higher k means that more relevant movies can be recommended.
# 
# ## Mean Average Precision (MAP)
# 
# Calculate the average precision for each user and average all the average precisions overall users. Penalizes incorrect rankings of movies.
# 
# ## Normalized Discounted Cumulative Gain (NDGC)
# 
# Looks at both relevant movies and the ranking order of the relevant movies.
# Normalized by the total number of users.

# In[30]:


# Create column with the predicted movie's rank for each user 
top_k = recommendations.copy()
top_k['rank'] = recommendations.groupby('userId', sort=False).cumcount() + 1  # For each user, only include movies recommendations that are also in the test set


# In[31]:


precision_at_k = metrics.precision_at_k(top_k, test, 'userId', 'movieId', 'rank')
recall_at_k = metrics.recall_at_k(top_k, test, 'userId', 'movieId', 'rank')
mean_average_precision = metrics.mean_average_precision(top_k, test, 'userId', 'movieId', 'rank')
ndcg = metrics.ndcg(top_k, test, 'userId', 'movieId', 'rank')


# In[32]:


print(f'Precision: {precision_at_k:.6f}',
      f'Recall: {recall_at_k:.6f}',
      f'MAP: {mean_average_precision:.6f} ',
      f'NDCG: {ndcg:.6f}', sep='\n')


# # Exploring movie embeddings
# 
# In this section, we examine how embeddings of movies relate to each other and if movies have similar movies near them in the embedding space. We will find the 6 closest movies to each movie. Remember that the closest movie should automatically be the same movie. Effectively we are finding the 5 closest films.
# 
# Here we find the movies that are closest to the movie 'Starwars' (movieId = 50). The closest movies are space-themed which makes complete sense, telling us that our movie embeddings are as intended. We also see this when looking at the closest movies for the kids' movie 'Lion King'.

# In[33]:


# Get the movie embeddings
_, new_item_embed = model(model.user_embedding, model.item_embedding)


# In[34]:


k = 6
nbrs = NearestNeighbors(n_neighbors=k).fit(new_item_embed)
distances, indices = nbrs.kneighbors(new_item_embed)

closest_movies = pd.DataFrame({
    'movie': np.repeat(np.arange(indices.shape[0])[:, None], k),
    'movieId': indices.flatten(),
    'distance': distances.flatten()
    }).replace({'movie': id2item,'movieId': id2item}).merge(movie_titles, how='left', on='movieId')
closest_movies


# In[35]:


id = 50
closest_movies[closest_movies.movie == id]


# In[36]:


id = 71
closest_movies[closest_movies.movie == id]

