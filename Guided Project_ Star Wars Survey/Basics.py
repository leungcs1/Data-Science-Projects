#!/usr/bin/env python
# coding: utf-8

# # Star Wars Survey

# In[1]:


import pandas as pd
star_wars = pd.read_csv('star_wars.csv', encoding='ISO-8859-1')


# In[2]:


star_wars = star_wars[pd.notnull(star_wars['RespondentID'])]


# In[3]:


star_wars.head(10)


# In[4]:


star_wars.columns


# In[5]:


yes_no = {'Yes': True, 'No': False}

for col in [
    "Have you seen any of the 6 films in the Star Wars franchise?",
    "Do you consider yourself to be a fan of the Star Wars film franchise?"]:
    star_wars[col] = star_wars[col].map(yes_no)
    
star_wars.head()


# In[6]:


import numpy as np

movie_mapping = {
    "Star Wars: Episode I  The Phantom Menace": True,
    np.nan: False,
    "Star Wars: Episode II  Attack of the Clones": True,
    "Star Wars: Episode III  Revenge of the Sith": True,
    "Star Wars: Episode IV  A New Hope": True,
    "Star Wars: Episode V The Empire Strikes Back": True,
    "Star Wars: Episode VI Return of the Jedi": True
}

for col in star_wars.columns[3:9]:
    star_wars[col] = star_wars[col].map(movie_mapping)


# In[7]:


star_wars = star_wars.rename(columns={
        "Which of the following Star Wars films have you seen? Please select all that apply.": "seen_1",
        "Unnamed: 4": "seen_2",
        "Unnamed: 5": "seen_3",
        "Unnamed: 6": "seen_4",
        "Unnamed: 7": "seen_5",
        "Unnamed: 8": "seen_6"
        })

star_wars.head()


# In[8]:


star_wars = star_wars.rename(columns={
        "Please rank the Star Wars films in order of preference with 1 being your favorite film in the franchise and 6 being your least favorite film.": "ranking_1",
        "Unnamed: 10": "ranking_2",
        "Unnamed: 11": "ranking_3",
        "Unnamed: 12": "ranking_4",
        "Unnamed: 13": "ranking_5",
        "Unnamed: 14": "ranking_6"
        })

star_wars.head()


# In[9]:


star_wars[star_wars.columns[9:15]] = star_wars[star_wars.columns[9:15]].astype(float)

star_wars[star_wars.columns[9:15]].mean()


# In[10]:


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

plt.bar(range(6), star_wars[star_wars.columns[9:15]].mean())


# # Rankings
# 
# So far, we've cleaned up the data, renamed several columns, and computed the average ranking of each movie. As I suspected, it looks like the "original" movies are rated much more highly than the newer ones.

# In[11]:


star_wars[star_wars.columns[3:9]].sum()


# In[12]:


plt.bar(range(6), star_wars[star_wars.columns[3:9]].sum())


# # View counts
# 
# It appears that the original movies were seen by more respondents than the newer movies. This reinforces what we saw in the rankings, where the earlier movies seem to be more popular.

# In[13]:


males = star_wars[star_wars["Gender"] == "Male"]
females = star_wars[star_wars["Gender"] == "Female"]


# In[14]:


plt.bar(range(6), males[males.columns[9:15]].mean())
plt.show()

plt.bar(range(6), females[females.columns[9:15]].mean())
plt.show()


# In[15]:


plt.bar(range(6), males[males.columns[3:9]].sum())
plt.show()

plt.bar(range(6), females[females.columns[3:9]].sum())
plt.show()


# # Male/Female differences in favorite Star Wars movie and most seen movie
# 
# Interestingly, more males watches episodes 1-3, but males liked them far less than females did.
