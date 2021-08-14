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


# ---

# # Exploratory Data Analysis
# 
# In this notebook we explore the MovieLens 100k dataset.
# 
# 
# *   Find missing/null values
# *   Examine the distribution of ratings
# *   Examine movies and users with most reviews
# *   Examine correlation between time and reviews
# 
# 

# # Imports

# In[13]:


import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import requests
import seaborn as sns
from scipy.stats.stats import pearsonr
from tqdm import tqdm


# # Prepare data

# In[16]:


# Load reviews.
fp = os.path.join('./data/bronze', 'u.data')
raw_data = pd.read_csv(fp, sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'])
raw_data.head()


# In[17]:


# Load movie titles.
fp = os.path.join('./data/bronze', 'u.item')
movie_titles = pd.read_csv(fp, sep='|', names=['movieId', 'title'], usecols = range(2), encoding='iso-8859-1')
movie_titles.head()


# In[18]:


# Merge dataframes.
raw_data = raw_data.merge(movie_titles, how='left', on='movieId')
raw_data.head()


# In[19]:


# Change timestamp to datatime.
raw_data.timestamp = pd.to_datetime(raw_data.timestamp, unit='s')
raw_data.head()


# # Exploration

# ## Unique and null values

# We first see that there are 100k observations in our dataset. There are 943 unique users and 1682 unique movies, and the rating system is out of 5. We then check to see if there are any missing data points in the set, which we find there are none.

# In[20]:


print(f'Shape: {raw_data.shape}')
raw_data.sample(5, random_state=123)


# In[21]:


raw_data.nunique()


# In[22]:


raw_data.info()


# In[23]:


print(f'Shape: {movie_titles.shape}')
movie_titles.sample(5, random_state=123)


# ## Summary Stats

# ### Ratings
# 
# Next, we look at the summary statistics of each feature in the dataset. We notice that the mean rating of the movies is 3.5 and that the minimum and maximum rating is 1 and 5 respectivle, and that the ratings are discrete (no in-between values). The most common rating is 4, with the second most common being 3. There are very few reviews with a 1 rating (about 6000/100,000). In fact looking at our boxplots, reviews where the movie is rated 1 might even be considered an outlier.

# In[24]:


raw_data.describe()


# In[25]:


plt.figure(figsize=(7,5))
sns.histplot(raw_data.rating)
plt.show()


# In[26]:


plt.figure(figsize=(10,6))
sns.boxplot(x = raw_data.rating)
plt.show()


# ### Time
# 
# Actual reviews were made starting from September 20, 1997 to April 22, 1998, about 7 months of data.
# 
# Actual movies reviewed were released from 1922 to 1998, with 4 years missing in that timespan. There are also a couple of movies with no year given. We assigned these movies to year 0.

# In[28]:


raw_data.timestamp.describe(datetime_is_numeric=True)


# In[29]:


def get_year(title):
    year=re.search(r'\(\d{4}\)', title)
    if year:
        year=year.group()
        return int(year[1:5])
    else:
        return 0


# In[30]:


raw_data['year'] = raw_data.title.apply(get_year)
raw_data.year.sort_values().unique()


# In[31]:


raw_data[['year']].nunique()


# In[32]:


sns.histplot(raw_data['year'][raw_data['year'] != 0])
plt.show()


# ## Users with most reviews
# 
# The most movies single user has reviewed is 737 reviews. The minimum number of reviews a user has reviewed in the dataset is 20. This is good since when creating recommendation systems, you want users with lots or reviews, allowing for us to test our recomendations. We also notice that most users reviewed less than 65 movies.

# In[33]:


users_count = raw_data.groupby('userId')['rating'].count().sort_values(ascending=False).reset_index()
users_count


# In[34]:


# Plot how many movies a user reviewed
plt.figure(figsize=(10, 6))
fig = sns.histplot(users_count['rating'])
plt.show()


# In[35]:


users_count['rating'].median()


# ## Movies with most reviews
# 
# As we can expect, popular movies such as 'Star Wars' and 'Toy Story' have the most reviews. The highest number of reviews is 583 while the lowest number of reviews is 1.

# In[36]:


movies_count = raw_data.groupby('title')['rating'].count().sort_values(ascending=False).reset_index()
movies_count


# In[37]:


# Plot 50 most reviewed movies.
plt.figure(figsize=(15,10))
fig = sns.barplot(x=movies_count.head(50)['title'], y=movies_count.head(50)['rating'])
fig.set_xticklabels(fig.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.tight_layout()
plt.show()


# ## Time correlation
# 
# Lastly we will examine if there is a correlation between then the movie was made and the rating given.

# ## Year movie released vs rating

# With a correlation coefficient of -0.1050, there is a tiny inverse relationship between when a movie was released and the rating given to it. The p-value is also much lower than 0.05 meaning that we can conclude that the correlation is statistically significant. Older movies were rating more generously than newer movies.
# 
# This could be because older movies do not have as many ratings as the newer movies. People who would actually watch and rate old movies from the 20s and 30s would typically be film enthusiasts and thus have a bias towards older movies.

# In[38]:


plt.figure(figsize=(10, 6))
mean_rating = raw_data.groupby('year')['rating'].mean().reset_index()
mean_rating = mean_rating[mean_rating.year != 0]
sns.lineplot(x=mean_rating.year, y=mean_rating.rating)
plt.ylabel('avg_rating')
plt.show()


# In[39]:


pearsonr(raw_data.year, raw_data.rating)

