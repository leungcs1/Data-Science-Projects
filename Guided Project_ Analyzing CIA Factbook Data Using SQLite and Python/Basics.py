#!/usr/bin/env python
# coding: utf-8

# In this project, we'll work with data from the [CIA World Factbook](https://www.cia.gov/library/publications/the-world-factbook/), a compendium of statistics about all of the countries on Earth. The Factbook contains demographic information like:
# 
# - population - The population as of 2015.
# - population_growth - The annual population growth rate, as a percentage.
# - area - The total land and water area.
# 
# You can download the SQLite database, factbook.db, [from this GitHub repo](https://github.com/factbook/factbook.sql/releases) if you want to work with it on your own computer. In this guided project, we'll explore the Python SQLite workflow to explore, analyze, and visualize data from this database.

# # Introduction

# In[1]:


import sqlite3
import pandas as pd

conn = sqlite3.connect("factbook.db")
cursor = conn.cursor()

q1 = "SELECT * FROM sqlite_master WHERE type='table';"

cursor.execute(q1).fetchall()


# In[2]:


pd.read_sql_query(q1, conn)


# In[3]:


q2 = "select * from facts limit 5"
pd.read_sql_query(q2, conn)


# # Summary Statistics

# In[4]:


q3 = 'select min(population) min_pop, max(population) max_pop, min(population_growth) min_pop_growth, max(population_growth) max_prop_growth from facts'

pd.read_sql_query(q3, conn)


# A few things stick out from the summary statistics in the last screen:
# 
# - there's a country with a population of 0
# - there's a country with a population of 7256490011 (or more than 7.2 billion people)
# 

# # Outliers

# In[5]:


q4 = 'select * from facts where population == (select max(population) from facts);'

pd.read_sql_query(q4, conn)


# In[6]:


q5 = 'select * from facts where population == (select min(population) from facts);'

pd.read_sql_query(q5, conn)


# It seems like the table contains a row for the whole world, which explains the population of over 7.2 billion. It also seems like the table contains a row for Antarctica, which explains the population of 0. This seems to match the CIA Factbook page for [Antarctica](https://www.cia.gov/library/publications/the-world-factbook/geos/ay.html)

# # Histograms

# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)

q6 = 'select population, population_growth, birth_rate, death_rate from facts where population != (select max(population) from facts) and population != (select min(population) from facts);'

pd.read_sql_query(q6, conn).hist(ax=ax)


# # Which countries have the highest population density?

# In[8]:


q7 = "select name, cast(population as float)/cast(area as float) density from facts order by density desc limit 20"
pd.read_sql_query(q7, conn)


# In[9]:


q7 = '''select population, population_growth, birth_rate, death_rate
from facts
where population != (select max(population) from facts)
and population != (select min(population) from facts);
'''
pd.read_sql_query(q7, conn)

