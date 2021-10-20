#!/usr/bin/env python
# coding: utf-8

# # Exercise 10: Hierarchical clustering of the grain data
# 
# In the video, you learnt that the SciPy `linkage()` function performs hierarchical clustering on an array of samples.  Use the `linkage()` function to obtain a hierarchical clustering of the grain samples, and use `dendrogram()` to visualize the result.  A sample of the grain measurements is provided in the array `samples`, while the variety of each grain sample is given by the list `varieties`.
# 

# **Step 1:** Import:
# 
#  + `linkage` and `dendrogram` from `scipy.cluster.hierarchy`.
#  + `matplotlib.pyplot` as `plt`.
#  + `pandas`
#  + `load_iris` and `train_test_split`

# In[2]:


import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import linkage, dendrogram

import matplotlib.pyplot as plt


# **Step 2:** Load the dataset _(done for you)_.

# In[20]:



iris_data = load_iris()
iris_data


# In[22]:


### BEGIN SOLUTION
#Create pd DF the same way you did the first day 
def create_df():
    df = pd.DataFrame(iris_data.data, columns=['sepal length','sepal width','petal length','petal width'])
    df['target']= iris_data.target
    df['class']= iris_data.target_names[iris_data.target]
    return df
#Create the class and target columns and remove de cm from the columns names

    
    # Add target and class to DataFrame
    
    #rename the columns - remove the (cm)
  
    
    
    ### END SOLUTION


# In[24]:


df_iris = create_df() 
df_iris.sample(n=10)  

assert df_iris['sepal length'].shape == (150,)
assert df_iris['sepal width'].shape == (150,)
assert df_iris['petal length'].shape == (150,)
assert df_iris['petal width'].shape == (150,)
assert df_iris['target'].shape == (150,)
assert df_iris['class'].shape == (150,)


# In[26]:



#From our data we will only get variable columns as np.array
x = 0

### BEGIN SOLUTION
x = df_iris.values[:,:4]

### END SOLUTIONS


# In[27]:


assert x.shape == (150,4)


# **Step 3:** Perform hierarchical clustering on `samples` using the `linkage()` function with the `method='complete'` keyword argument. Assign the result to `mergings`.

# In[28]:


mergings = 0 
### BEGIN SOLUTION
mergings = linkage(x, 'complete')

### END SOLUTION


# In[ ]:


assert mergings.shape == (149,4)


# **Step 4:** Plot a dendrogram using the `dendrogram()` function on `mergings`, specifying the keyword arguments `labels=varieties`, `leaf_rotation=90`, and `leaf_font_size=6`.  Remember to call `plt.show()` afterwards, to display your plot.

# In[34]:


dn = 0
### BEGIN SOLUTION
fig = plt.figure(figsize= (40,10))
dn = dendrogram(mergings, leaf_rotation=90, leaf_font_size=6)

# END SOLUTION


# In[35]:


assert type(dn) == type(dendrogram(np.random.rand(15, 4)))


# ## K-MEANS
# 
# for this next part we will use the K-Means algorithm in order to cluster your data

# **Step 1:** 
# + import `KMeans` from `sklearn.cluster`
# + loading our new datasets

# In[38]:


from sklearn.cluster import KMeans
df = pd.read_csv('ch1ex1.csv')
points = df.values

new_df = pd.read_csv('ch1ex2.csv')
new_points = new_df.values


# **Step 2:** Using `KMeans()`, create a `KMeans` instance called `model` to find `3` clusters. To specify the number of clusters, use the `n_clusters` keyword argument
# 

# In[46]:


from sklearn.cluster import KMeans
model = 0
### BEGIN SOLUTION

model = KMeans( n_clusters=3 )
model

### END SOLUTION


# In[40]:


assert type(model)== type(KMeans())


# **Step 4:** Use the `.fit()` method of `model` to fit the model to the array of points `points`.

# In[47]:


### BEGIN SOLUTION

model= model.fit(points)
model

### END SOLUTION


# **Step 5:** Use the `.predict()` method of `model` to predict the cluster labels of `points`, assigning the result to `labels`.

# In[48]:


### BEGIN SOLUTION

labels= model.predict(points)

### END SOLUTION


# In[ ]:


assert labels[labels.argmax()] == 2
assert labels.shape == (300,)


# In[49]:


# Make a function that returns 3 numpy arrays each one with the points associated for each class
#If the label is 0 they go into data_0
#If the label is 1 they go into data_1
#If the label is 2 they go into data_2

def separate_labels(labels, points):
    data_0 = [ labels==0 ]
    data_1 = [ labels==1 ]
    data_2 = [ labels==2 ]
    return data_0,data_1,data_2
data_0,data_1,data_2 = separate_labels(labels, points)
    ### BEGIN SOLUTION

    
    ### END SOLUTION


# In[ ]:


assert abs(data_0.shape[0] - 94) <= 20
assert abs(data_1.shape[0] - 95) <= 20
assert abs(data_2.shape[0] - 111) <= 20


# In[6]:


# plotting the data 

### BEGIN SOLUTION

plt.scatter(data_0[:,0] , data_0[:,1] , color = 'blue')
plt.scatter(data_1[:,0] , data_1[:,1] , color = 'red')
plt.scatter(data_2[:,0] , data_2[:,1] , color = 'yellow')

### END SOLUTION


# **Step 7:** Use the `.predict()` method of `model` to predict the cluster labels of `new_points`, assigning the result to `new_labels`.  Notice that KMeans can assign previously unseen points to the clusters it has already found!

# In[50]:



### BEGIN SOLUTION

new_labels = model.predict(new_points)

### END SOLUTION


# In[ ]:


assert new_labels[new_labels.argmax()] == 2
assert new_labels.shape == (100,)


# In[51]:


#separate the data by labels like we did before

new_0 = []
new_1 = []
new_2 = []

### BEGIN SOLUTION

new_0, new_1, new_2 = separate_labels(new_labels, new_points)

### END SOLUTION


# In[ ]:


assert abs(new_0.shape[0] - 34) <= 5
assert abs(new_1.shape[0] - 34) <= 5
assert abs(new_2.shape[0] - 34) <= 5


# In[4]:


#plot the new labels


### BEGIN SOLUTION

plt.scatter(new_0[:,0] , new_0[:,1] , color = 'yellow')
plt.scatter(new_1[:,0] , new_1[:,1] , color = 'red')
plt.scatter(new_2[:,0] , new_2[:,1] , color = 'green')

### END SOLUTION


# In[5]:


#plot all the points together

### BEGIN SOLUTION


plt.scatter(data_0[:,0] , data_0[:,1] , color = 'blue')
plt.scatter(data_1[:,0] , data_1[:,1] , color = 'red')
plt.scatter(data_2[:,0] , data_2[:,1] , color = 'yellow')

plt.scatter(new_0[:,0] , new_0[:,1] , color = 'yellow')
plt.scatter(new_1[:,0] , new_1[:,1] , color = 'red')
plt.scatter(new_2[:,0] , new_2[:,1] , color = 'green')

plt.show()

### END SOLUTION


# In[ ]:




