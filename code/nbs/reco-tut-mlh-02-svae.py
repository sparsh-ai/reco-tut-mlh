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


# In[36]:


get_ipython().system(u'git status')


# In[38]:


get_ipython().system(u'git pull --rebase origin main')


# In[39]:


get_ipython().system(u'git add . && git commit -m \'commit\' && git push origin "{branch}"')


# In[3]:


import sys
sys.path.insert(0,'./code')


# ---

# # Standard Variational Autoencoder (SVAE)
# 
# The Standard Variational Autoencoder (SVAE), SVAE uses an autoencoder to generate a salient feature representation of users, learning a latent vector for each user. The decoder then takes this latent representation and outputs a probability distribution over all items; we get probabilities of all the movies being watched by each user.

# # Imports

# In[27]:


import numpy as np
import os
import pandas as pd

from utils import numpy_stratified_split
import build_features
import metrics
from models import SVAE
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()


# # Prepare Data

# In[6]:


fp = os.path.join('./data/bronze', 'u.data')
raw_data = pd.read_csv(fp, sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'])
print(f'Shape: {raw_data.shape}')
raw_data.sample(5, random_state=123)


# In[7]:


# Binarize the data (only keep ratings >= 4)
df_preferred = raw_data[raw_data['rating'] > 3.5]
print (df_preferred.shape)
df_low_rating = raw_data[raw_data['rating'] <= 3.5]

df_preferred.head(10)


# In[8]:


# Keep users who clicked on at least 5 movies
df = df_preferred.groupby('userId').filter(lambda x: len(x) >= 5)

# Keep movies that were clicked on by at least on 1 user
df = df.groupby('movieId').filter(lambda x: len(x) >= 1)

print(df.shape)


# In[9]:


# Obtain both usercount and itemcount after filtering
usercount = df[['userId']].groupby('userId', as_index = False).size()
itemcount = df[['movieId']].groupby('movieId', as_index = False).size()

# Compute sparsity after filtering
sparsity = 1. * raw_data.shape[0] / (usercount.shape[0] * itemcount.shape[0])

print("After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)" % 
      (raw_data.shape[0], usercount.shape[0], itemcount.shape[0], sparsity * 100))


# ## Split

# In[10]:


unique_users =sorted(df.userId.unique())
np.random.seed(123)
unique_users = np.random.permutation(unique_users)


# In[11]:


HELDOUT_USERS = 200

# Create train/validation/test users
n_users = len(unique_users)
print("Number of unique users:", n_users)

train_users = unique_users[:(n_users - HELDOUT_USERS * 2)]
print("\nNumber of training users:", len(train_users))

val_users = unique_users[(n_users - HELDOUT_USERS * 2) : (n_users - HELDOUT_USERS)]
print("\nNumber of validation users:", len(val_users))

test_users = unique_users[(n_users - HELDOUT_USERS):]
print("\nNumber of test users:", len(test_users))


# In[12]:


# For training set keep only users that are in train_users list
train_set = df.loc[df['userId'].isin(train_users)]
print("Number of training observations: ", train_set.shape[0])

# For validation set keep only users that are in val_users list
val_set = df.loc[df['userId'].isin(val_users)]
print("\nNumber of validation observations: ", val_set.shape[0])

# For test set keep only users that are in test_users list
test_set = df.loc[df['userId'].isin(test_users)]
print("\nNumber of test observations: ", test_set.shape[0])

# train_set/val_set/test_set contain user - movie interactions with rating 4 or 5


# In[13]:


# Obtain list of unique movies used in training set
unique_train_items = pd.unique(train_set['movieId'])
print("Number of unique movies that rated in training set", unique_train_items.size)


# In[14]:


# For validation set keep only movies that used in training set
val_set = val_set.loc[val_set['movieId'].isin(unique_train_items)]
print("Number of validation observations after filtering: ", val_set.shape[0])

# For test set keep only movies that used in training set
test_set = test_set.loc[test_set['movieId'].isin(unique_train_items)]
print("\nNumber of test observations after filtering: ", test_set.shape[0])


# In[16]:


# Instantiate the sparse matrix generation for train, validation and test sets
# use list of unique items from training set for all sets
am_train = build_features.AffinityMatrix(df=train_set, items_list=unique_train_items)

am_val = build_features.AffinityMatrix(df=val_set, items_list=unique_train_items)

am_test = build_features.AffinityMatrix(df=test_set, items_list=unique_train_items)


# In[17]:


# Obtain the sparse matrix for train, validation and test sets
train_data, _, _ = am_train.gen_affinity_matrix()
print(train_data.shape)

val_data, val_map_users, val_map_items = am_val.gen_affinity_matrix()
print(val_data.shape)

test_data, test_map_users, test_map_items = am_test.gen_affinity_matrix()
print(test_data.shape)


# In[21]:


# Split validation and test data into training and testing parts
val_data_tr, val_data_te = numpy_stratified_split(val_data, ratio=0.75, seed=123)
test_data_tr, test_data_te = numpy_stratified_split(test_data, ratio=0.75, seed=123)


# In[22]:


# Binarize train, validation and test data
train_data = np.where(train_data > 3.5, 1.0, 0.0)
val_data = np.where(val_data > 3.5, 1.0, 0.0)
test_data = np.where(test_data > 3.5, 1.0, 0.0)

# Binarize validation data: training part  
val_data_tr = np.where(val_data_tr > 3.5, 1.0, 0.0)
# Binarize validation data: testing part (save non-binary version in the separate object, will be used for calculating NDCG)
val_data_te_ratings = val_data_te.copy()
val_data_te = np.where(val_data_te > 3.5, 1.0, 0.0)

# Binarize test data: training part 
test_data_tr = np.where(test_data_tr > 3.5, 1.0, 0.0)

# Binarize test data: testing part (save non-binary version in the separate object, will be used for calculating NDCG)
test_data_te_ratings = test_data_te.copy()
test_data_te = np.where(test_data_te > 3.5, 1.0, 0.0)


# In[23]:


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


# # SVAE

# In[29]:


INTERMEDIATE_DIM = 200
LATENT_DIM = 64
EPOCHS = 400
BATCH_SIZE = 100


# In[30]:


model = SVAE.StandardVAE(n_users=train_data.shape[0], # Number of unique users in the training set
                                   original_dim=train_data.shape[1], # Number of unique items in the training set
                                   intermediate_dim=INTERMEDIATE_DIM, 
                                   latent_dim=LATENT_DIM, 
                                   n_epochs=EPOCHS, 
                                   batch_size=BATCH_SIZE, 
                                   k=10,
                                   verbose=0,
                                   seed=123,
                                   drop_encoder=0.5,
                                   drop_decoder=0.5,
                                   annealing=False,
                                   beta=1.0
                                   )


# In[31]:


get_ipython().run_cell_magic(u'time', u'', u'model.fit(x_train=train_data,\n          x_valid=val_data,\n          x_val_tr=val_data_tr,\n          x_val_te=val_data_te_ratings, # with the original ratings\n          mapper=am_val\n          )')


# # Recommend

# In[32]:


# Model prediction on the training part of test set 
top_k =  model.recommend_k_items(x=test_data_tr,k=10,remove_seen=True)

# Convert sparse matrix back to df
recommendations = am_test.map_back_sparse(top_k, kind='prediction')
test_df = am_test.map_back_sparse(test_data_te_ratings, kind='ratings') # use test_data_te_, with the original ratings


# ## Evaluation metrics

# In[33]:


# Create column with the predicted movie's rank for each user 
top_k = recommendations.copy()
top_k['rank'] = recommendations.groupby('userId', sort=False).cumcount() + 1  # For each user, only include movies recommendations that are also in the test set


# In[34]:


precision_at_k = metrics.precision_at_k(top_k, test_df, 'userId', 'movieId', 'rank')
recall_at_k = metrics.recall_at_k(top_k, test_df, 'userId', 'movieId', 'rank')
mean_average_precision = metrics.mean_average_precision(top_k, test_df, 'userId', 'movieId', 'rank')
ndcg = metrics.ndcg(top_k, test_df, 'userId', 'movieId', 'rank')


# In[35]:


print(f'Precision: {precision_at_k:.6f}',
      f'Recall: {recall_at_k:.6f}',
      f'MAP: {mean_average_precision:.6f} ',
      f'NDCG: {ndcg:.6f}', sep='\n')


# # References
# 
# 
# 1.   Kilol Gupta, Mukunds Y. Raghuprasad, Pankhuri Kumar, A Hybrid Variational Autoencoder for Collaborative Filtering, 2018, https://arxiv.org/pdf/1808.01006.pdf
# 
# 2.   Microsoft SVAE implementation: https://github.com/microsoft/recommenders/blob/main/examples/02_model_collaborative_filtering/standard_vae_deep_dive.ipynb
# 
