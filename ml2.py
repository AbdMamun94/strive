#!/usr/bin/env python
# coding: utf-8

# # Linear Regression From Scratch
# In this notebook you will implement linear regression using the least squares method.
# 
# The notebook will guide you in the general steps. You may use only numpy, pandas, and matplotlib for this exercise.
# 
# #### Math Background
# The task is to solve linear regression for the data set of ```x``` and ```y```. That is, fit a line y to the data such that ```y = a + bx + e```. Where a and b are coefficents and e is an error term.
# 
# We know that ```b = SUM ( xi - x_mean) * (yi - y_mean) / SUM (xi - x_mean)^2``` where ```xi```, and ```yi``` are the indivdual observations and ```x_mean```, ```y_mean``` are means of all ```xi``` and ```yi```.
# 
# So a reasonable solution then for a fit is ```a = y_mean - b * x_mean```.
# 
# 
# #### Implementation Steps
# 1. Load the file ```reg_data.csv```
# 2. Display the data with a scatter plot. Write a markdown cell and comment on the structure of the data and what kind of line you think will fit the data. 
# 3. Implement a function of the least squares method.
# 4. Plot the predictions from your function.
# 5. Comment on the output of your function. Does it make sense? Can it be made better? If yes, how?

# In[3]:


#import the minimum packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[4]:


#load and inspect the data
df = 0
# YOUR CODE HERE
#raise NotImplementedError()


df = pd.read_csv('reg_data.csv')

df


# In[ ]:


assert df.shape == (100,2)


# In[5]:


#plot the data (scatter)
# YOUR CODE HERE
#raise NotImplementedError()

X = df['X']
y = df['Y']
plt.scatter(X, y)


# In[6]:


# YOUR CODE HERE
#raise NotImplementedError()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
random_state = 0)


# In[7]:


assert x_train.shape == (80,)
assert x_test.shape == (20,)
assert y_train.shape == (80,)
assert y_test.shape == (20,)


# In[13]:


#implement a least squares function to find a, b

# x_mean = 0
# y_mean = 0
# b_upper = 0
# b_lower = 0
# b = 0
# a = 0

# YOUR CODE HERE
#raise NotImplementedError()


x_mean = np.mean(x_train)
y_mean = np.mean(y_train)

A = len(x_train)

# Calculating a and b using formula
b_upper = 0
b_lower = 0

for i in range(A):
    b_upper += (x_train.values[i] - x_mean) * (y_train.values[i] - y_mean)
    b_lower += (x_train.values[i] - x_mean) ** 2

b = b_upper / b_lower
a = y_mean - b * x_mean



print(b)
print(a)


# In[14]:


assert abs(a - 7.72) <=0.03
assert abs(b - 1.32) <=0.03


# In[12]:


line = 0
x = np.array([])
# YOUR CODE HERE
#raise NotImplementedError()

line = 0
x = np.arange(0, 100)
line = a + b * x
#Plot
plt.plot(x,line, color="green", label="Regression Line")

plt.scatter(x_train,y_train,color = "blue", label="Scatter Plot")

plt.legend()
plt.show


# In[15]:


assert abs(a +3*b - 11.69) <= 0.05
assert abs(a +100*b - 140) <= 1


# In[ ]:


# YOUR CODE HERE
raise NotImplementedError()


# In[17]:


#Classify your test data in to classes
#if the Y value for a certain X is lower than the line then the class is 0
# class_0 = []
# class_1 = []

# # YOUR CODE HERE
# raise NotImplementedError()
        
# class_0 = np.array(class_0)
# class_1 = np.array(class_1)
# print(class_0.shape)
# print(class_1.shape)


class_0 = x_test[y_test<(a + b * x_test)]
class_1 = x_test[y_test>=(a + b * x_test)]

        
class_0 = np.array(class_0)
class_1 = np.array(class_1)
print(class_0.shape)
print(class_1.shape)
class_0


# In[ ]:


assert 9 < class_0.shape[0] < 13
assert 7 < class_1.shape[0] < 11


# In[ ]:


# YOUR CODE HERE
raise NotImplementedError()

