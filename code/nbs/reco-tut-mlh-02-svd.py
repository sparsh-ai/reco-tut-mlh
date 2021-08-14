#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[ ]:


get_ipython().system(u'git status')


# In[ ]:


get_ipython().system(u'git add . && git commit -m \'commit\' && git push origin "{branch}"')


# In[4]:


import sys
sys.path.insert(0, './code')


# ---

# # Singular Value Decomposition (SVD & SVD++)
# 
# SVD was heavily used in Netflix's Prize Competition in 2009. The grand prize of $1,000,000 was won by BellKor's Pragmatic Chaos. SVD utilizes stochastic gradient descent to attempt to decompose the original sparse matrices into lower ranking user and item factors (matrix factorization). These two matrices are then multiplied together to predict unknown values in the original sparse martix.
# 
# SVD++ adds a new  factor, the effect of implicit information instead of just the explicit information.

# # Imports

# In[6]:


get_ipython().system(u'pip install -q surprise')


# In[7]:


import os
import pandas as pd
import surprise

from utils import stratified_split
import metrics


# # Prepare data

# ## Load data

# In[8]:


fp = os.path.join('./data/bronze', 'u.data')
raw_data = pd.read_csv(fp, sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'])
print(f'Shape: {raw_data.shape}')
raw_data.sample(10, random_state=123)


# ## Train test split

# In[10]:


train_size = 0.75
train, test = stratified_split(raw_data, 'userId', train_size)

print(f'Train Shape: {train.shape}')
print(f'Test Shape: {test.shape}')
print(f'Do they have the same users?: {set(train.userId) == set(test.userId)}')


# # SVD and SVD++

# In[11]:


# Drop 'Timestamp' because surprise only takes dataframes with 3 columns in this order: userid, itemid, rating.
surprise_train = surprise.Dataset.load_from_df(train.drop('timestamp', axis=1), reader=surprise.Reader('ml-100k')).build_full_trainset()

# Instantiate models.
svd = surprise.SVD(random_state=0, n_factors=64, n_epochs=10, verbose=True)
svdpp = surprise.SVDpp(random_state=0, n_factors=64, n_epochs=10, verbose=True)
models = [svd, svdpp]

# Fit.
for model in models:
    model.fit(surprise_train)


# ## Recommend

# In[12]:


all_preds = []
for model in models:
    # Predict ratings for ALL movies for all users
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
    
    all_preds.append(merged)


# In[13]:


recommendations = []
for predictions in all_preds:
    # Create filter for users that appear in both the train and test set
    common_users = set(test['userId']).intersection(set(predictions['userId']))
    
    # Filter the test and predictions so they have the same users between them
    test_common = test[test['userId'].isin(common_users)]
    svd_pred_common = predictions[predictions['userId'].isin(common_users)]
    
    if len(set(predictions['userId'])) != len(set(test['userId'])):
        print('Number of users in train and test are NOT equal')
        print(f"# of users in train and test respectively: {len(set(predictions['userId']))}, {len(set(test['userId']))}")
        print(f"# of users in BOTH train and test: {len(set(svd_pred_common['userId']))}")
        continue
        
    # From the predictions, we want only the top k for each user,
    # not all the recommendations.
    # Extract the top k recommendations from the predictions
    top_movies = svd_pred_common.groupby('userId', as_index=False).apply(lambda x: x.nlargest(10, 'prediction')).reset_index(drop=True)
    top_movies['rank'] = top_movies.groupby('userId', sort=False).cumcount() + 1
    
    recommendations.append(top_movies)


# # Evaluation metrics
# 
# We see how SVD++ performs better than normal SVD in all metrics.

# In[14]:


model_metrics = {'svd':{}, 'svd++':{}}
for recommendation, model in zip(recommendations, model_metrics):
    # Create column with the predicted movie's rank for each user.
    top_k = recommendation.copy()
    top_k['rank'] = recommendation.groupby('userId', sort=False).cumcount() + 1  # For each user, only include movies recommendations that are also in the test set
    
    # Metrics.
    precision_at_k = metrics.precision_at_k(top_k, test, 'userId', 'movieId', 'rank')
    recall_at_k = metrics.recall_at_k(top_k, test, 'userId', 'movieId', 'rank')
    mean_average_precision = metrics.mean_average_precision(top_k, test, 'userId', 'movieId', 'rank')
    ndcg = metrics.ndcg(top_k, test, 'userId', 'movieId', 'rank')

    model_metrics[model]['precision'] = precision_at_k
    model_metrics[model]['recall'] = recall_at_k
    model_metrics[model]['MAP'] = mean_average_precision
    model_metrics[model]['NDCG'] = ndcg


# In[15]:


for model, values in model_metrics.items():
    print(f'------ {model} -------',
          f'Precision: {values["precision"]:.6f}',
          f'Recall: {values["recall"]:.6f}',
          f'MAP: {values["MAP"]:.6f} ',
          f'NDCG: {values["NDCG"]:.6f}',
          '', sep='\n')

