#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# We'll be working with a dataset on the job outcomes of students who graduated from college between 2010 and 2012. The original data on job outcomes was released by [American Community Survey](https://www.census.gov/programs-surveys/acs/), which conducts surveys and aggregates the data. FiveThirtyEight cleaned the dataset and released it on their [Github repo](https://github.com/fivethirtyeight/data/tree/master/college-majors).
# 
# Each row in the dataset represents a different major in college and contains information on gender diversity, employment rates, median salaries, and more.
# 
# Using visualizations, we can start to explore questions from the dataset like:
# 
# - Do students in more popular majors make more money?
#      - Using scatter plots
# - How many majors are predominantly male? Predominantly female?
#      - Using histograms
# - Which category of majors have the most students?
#      - Using bar plots
# 
# Before we start creating data visualizations, let's import the libraries we need and remove rows containing null values.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
recent_grads = pd.read_csv("recent-grads.csv")
print(recent_grads.iloc[0])
print(recent_grads.head())
print(recent_grads.tail())
recent_grads.describe()
recent_grads = recent_grads.dropna()


# # Pandas, Scatter Plots
# 
# We will generate scatter plots in separate cells to explore the following relations:
# 
# - Sample_size and Median
# - Sample_size and Unemployment_rate
# - Full_time and Median
# - ShareWomen and Unemployment_rate
# - Men and Median
# - Women and Median

# In[2]:


recent_grads.plot( x="Sample_size", y="Median", kind="scatter")


# In[3]:


recent_grads.plot( x="Sample_size", y="Unemployment_rate", kind="scatter")


# In[4]:


recent_grads.plot( x="Full_time", y="Median", kind="scatter")


# In[5]:


recent_grads.plot( x="ShareWomen", y="Unemployment_rate", kind="scatter")


# In[6]:


recent_grads.plot( x="Men", y="Median", kind="scatter")


# In[7]:


recent_grads.plot( x="Women", y="Median", kind="scatter")


# # Pandas, Histograms
# 
# We will generate histograms to explore the distributions of the following columns:
# 
# - Sample_size
# - Median
# - Employed
# - Full_time
# - ShareWomen
# - Unemployment_rate
# - Men
# - Women

# In[8]:


cols = ["Sample_size", "Median", "Employed", "Full_time", "ShareWomen", "Unemployment_rate", "Men", "Women"]

fig = plt.figure(figsize=(5,12))
for r in range(1,5):
    ax = fig.add_subplot(4,1,r)
    ax = recent_grads[cols[r]].plot(kind="hist", rot=40)


# In[10]:


cols = ["Sample_size", "Median", "Employed", "Full_time", "ShareWomen", "Unemployment_rate", "Men", "Women"]

fig = plt.figure(figsize=(5,12))
for r in range(4,8):
    ax = fig.add_subplot(4,1,r-3)
    ax = recent_grads[cols[r]].plot(kind="hist", rot=40)


# # Pandas, Scatter Matrix Plot

# In[11]:


from pandas.plotting import scatter_matrix
scatter_matrix(recent_grads[["Sample_size", "Median"]], figsize =(6,6))


# In[12]:


from pandas.plotting import scatter_matrix
scatter_matrix(recent_grads[["Sample_size", "Median", "Unemployment_rate"]], figsize =(10,10))


# # Pandas, Bar Plots

# First, we compare the percentages of women (ShareWomen) from the first ten rows and last ten rows of the recent_grads dataframe.

# In[14]:


recent_grads[:10].plot.bar(x="Major", y="ShareWomen", legend=False)
recent_grads[-10:].plot.bar(x="Major", y="ShareWomen", legend=False)


# Then we compare the unemployment rate (Unemployment_rate) from the first ten rows and last ten rows of the recent_grads dataframe.

# In[15]:


recent_grads[:10].plot.bar(x="Major", y="Unemployment_rate", legend=False)
recent_grads[-10:].plot.bar(x="Major", y="Unemployment_rate", legend=False)

