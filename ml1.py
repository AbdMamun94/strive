#!/usr/bin/env python
# coding: utf-8

# <div style="background:#222222; color:#ffffff; padding:20px">
# <h1 align="center">Guided ML With The Iris Dataset</h1>
# 
# <h2 align="center" tyle="color:#01ff84" >Learning type | Activity type | Objective |</h2>
# <h2 align="center">| Supervised | Multiclass classification | Identify a flower's class |</h2>
# 
# 
# <div>

# Contents:
# 
# 1. Loading the data
# 2. Setting up supervised learning problem (selecting features)
# 3. Creating a first model
#     - Creating train and test datasets
#     - Normalizing train and test
#     - Fitting and predicting
# 4. Evaluate the frist model predictions
# 5. Crossvalidation of the model
# 6. Creating an end to end ML pipeline
#     - Train/Test Split
#     - Normalize
#     - Crossvalidations
#     - Model
#     - fitting and predicting

# ## Instructions with NBGrader removed
# 
# Complete the cells beginning with `# YOUR CODE HERE` and run the subsequent cells to check your code.

# 
# Contents:
# 1. Loading the data
# 2. Setting up supervised learning problem (selecting features)
# 3. Creating a first model
#     - Creating train and test datasets
#     - Normalizing train and test
#     - Fitting and predicting
# 4. Evaluate the frist model predictions
# 5. Crossvalidation of the model
# 6. Creating an end to end ML pipeline
#     - Train/Test Split
#     - Normalize
#     - Crossvalidations
#     - Model
#     - fitting and predicting

# ## About the dataset
# 
# [Iris](https://archive.ics.uci.edu/ml/datasets/iris) is a well-known multiclass dataset. It contains 3 classes of flowers with 50 examples each. There are a total of 4 features for each flower.
# 
# ![](./classic-datasets/images/Iris-versicolor-21_1.jpg)

# ## Package setups
# 
# 1. Run the following two cells to initalize the required libraries. 

# In[15]:


#to debug package errors
import sys
sys.path
sys.executable


# In[16]:


# Import needed packages
# You may add or remove packages should you need them
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import datasets
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import make_pipeline

# Display plots inline and change plot resolution to retina
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
# Set Seaborn aesthetic parameters to defaults
sns.set()


# In[ ]:





# ## Step 1: Loading the data
# 
# 1. Load the iris dataset using ```datasets.load_iris()```
# 2. Investigate the data structure with ```.keys()```
# 3. Construct a dataframe from the dataset
# 4. Create a 'target' and a 'class' column that contains the target names and values
# 5. Display a random sample of the dataframe 

# In[17]:


def load_data():
    #load the dataset
    #return the dataset
    # YOUR CODE HERE
    data = datasets.load_iris()
    
    #raise NotImplementedError()
    
    return data


# In[18]:


assert load_data()['data'].shape == (150,4)


# In[19]:


dataset = load_data()
load_data().feature_names


# In[20]:


dataset


# In[21]:


dataset.target


# In[22]:


def dataset_to_pandas():
    #put the dataset into a pandas DF using the feature names as columnsç
    #rename the column name so the dont include the '(cm)'
    #add 2 columns one with the target and another with the target_names
    df = pd.DataFrame(dataset.data,columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
    df['target']=dataset.target
    df['class']=dataset.target_names[dataset.target]
    #df['target_names']=load_data().target_names
    return df
   
    raise NotImplementedError()


# In[23]:


df = dataset_to_pandas()
assert df['sepal length'].shape == (150,)
assert df['sepal width'].shape == (150,)
assert df['petal length'].shape == (150,)
assert df['petal width'].shape == (150,)
assert df['target'].shape == (150,)
assert df['class'].shape == (150,)


# In[24]:


df


# ### Question
# Find the X and y values we're looking for. Notice that y is categorical and thus, we could **one-hot encode it** if we are looking at **class** or we can just pick **target**. In order to one hot encode we have  to re-shape `y` it using the **.get_dummies** function. 
# 
# ## For the purpose of this exercise, do not use hot encoding, go only for target but think about if you have to drop it somewhere or not...

# In[28]:


df_iris = dataset_to_pandas()


# In[29]:


# virginica close to versicolor BAD
#0 - setosa
#1 - versicolor
#2 - virginica 

# Same distance between all the classes GOOD
#[1,0,0] - setosa
#[0,1,0] - versicolor
#[0,0,1] - virginica 


def ohe():
    ohe_data=pd.get_dummies(df_iris,prefix="data")
    
    return ohe_data

    # YOUR CODE HERE
    #raise NotImplementedError()


# In[30]:


ohe()


# In[32]:


ohe_data = ohe()

assert ohe_data.shape == (150,8)


# ## Step 2: Setting up supervised learning problem (selecting features)
# 
# Feature selection is an essential step in improving a model's perfromance. In the first version of the model we will use the **'sepal length'** and **'sepal width'** as predicting features. Later we will see the effect of adding additional features.
# 
# 1. Assign the values of the 'target' to Y as a numpy array
# 2. Assign the remaining feature values to X as a numpy array
# 3. Check the shape of X and Y. Check the first few values.
#     - Can we confirm our X and Y are created correctly?

# In[33]:


def target_to_numpy():
    # YOUR CODE HERE
    y = ohe_data["target"].to_numpy()

    return y
    
    #raise NotImplementedError()
def data_to_numpy():
    # YOUR CODE HERE
    x = ohe_data[['sepal length','sepal width']].to_numpy()
    return x
    
    #raise NotImplementedError()


# In[34]:


Y = target_to_numpy()
X = data_to_numpy()
assert isinstance(Y, np.ndarray)
assert isinstance(X, np.ndarray)
assert X.shape == (150,2)


# In[35]:


#your code here
X = df_iris[['sepal length', 'sepal width']].values
print(X.shape)
X[:5]


# ## Step 3: Creating the first model
# 
# In lecture we learned about creating a train and test datasets, normalizing, and fitting a model. In this step we will see how to build a simple version of this.
# 
# We have to be careful when constructing our train and test datasets. First, when we create train and test datasets we have to be careful that we always have the same datapoints in each set. Otherwise our results won't be reproduceable or we might introduce a bias into our model.
# 
# We also need to be attentive to when we normalize the data. What would be the effect of normalizing the data (i.e. with StandardScaler to a range between 0 - 1) before we create our train and test sets? Effectively we would use information in the test set to structure the values in the training set and vice versa. Therefore normalizing train and test independently is the preferred method.
# 
# 1. Create X_train, X_test, Y_train, Y_test using ```train_test_split()``` with an 80/20 train/test split. Look in the SKLearn documentation to understand how the function works.
#     - Inspect the first few rows of X_train.
#     - Run the cell a few times. Do the first few rows change?
#     - What option can we use in ```train_test_split()``` to stop this from happening?
# 2. Normalize the train and test datasets with ```StandardScaler```
#     - We can fit the transform with ```.fit()``` and ```.transform()``` to apply it. Look in the documentation for an esample of how to do this.
#     - Does it make sense to normalize Y_train and Y_test?
# 3. Initalize a ```LogisticRegression()``` model and use the ```.fit()``` method to initalize the first model.
#     - We will pass the X_train and Y_train variables to the ```.fit()``` method.
#     - Once the model is fit, use the ```.predict()``` with the X_test and save the output as predictions.

# In[37]:


#split train and test data 80/20
#your code here
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)

# YOUR CODE HERE
#raise NotImplementedError()

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# In[38]:


X_train


# In[49]:


assert X_train.shape == (120,2)
assert Y_train.shape == (120,)
assert X_test.shape  == (30,2)
assert Y_test.shape  == (30,)


# In[50]:


#normalize the dataset
#your code here
scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#create and fit the scaler object on the training data
# YOUR CODE HERE
#raise NotImplementedError()

X_train[:5]


# In[51]:


Y_train[:5]


# In[52]:


assert np.amin(X_train) >= -2.5
assert np.amax(X_train) <= 3.2
assert np.amin(X_test) >= -2
assert np.amin(X_test) <= 2.75


# In[53]:


#initalize and fit with Logistic Regression
#prediction = 0
#your code here
model_standard = LogisticRegression()

model_standard.fit(X_train,Y_train)

predictions = model_standard.predict(X_test)

#initalize the logistic regressor
#make predictions
# YOUR CODE HERE
#raise NotImplementedError()


# In[54]:


assert predictions.shape == (30,)


# ## Step 4: Evaluate the frist model's predictions
# 
# We will learn more about how to evaluate the performance of a classifier in later lessons. For now we will use % accuracy as our metric. It is important to know that this metric only helps us understand the specific performance of our model and not, for example, where we can improve it, or where it already perfoms well.
# 
# 1. Use ```.score()``` to evaluate the performance of our first model.

# In[55]:


#score = 0
#evaluating the performace of our first model
#your code here
# YOUR CODE HERE
score = accuracy_score(Y_test,predictions)
#raise NotImplementedError()
score


# In[ ]:


assert score >=0.7


# In[ ]:


assert score >=0.72


# In[ ]:


assert score >=0.73


# ## Step 5: Crossvalidation of the model
# Our first model achived ~90% accruacy. This is quite good. How do we know it is reproducable? If we run the model again and our performance is 85% which is correct? And what about improving our model? Can you think of one thing we can do to potentially improve the model?
# 
# #### Crossvalidation
# Corssvalidation is when we create multiple X and Y datasets. On each dataset we train and fit the model. We then average the results and return a 'crossvalidated' accruacy.
# 
# 1. Initalize a new version of the model you trained above with the same paramters.
# 2. Use ```cross_validate()``` to run the model with 5 crossvalidation folds. 

# In[57]:


#model with cross validation
#your code here
model_cross = LogisticRegression()
cv = cross_validate(model_cross,X_train,Y_train,cv=5)
#cross validate the training set
#clf_cv = 0
#CV = 0
# YOUR CODE HERE

#raise NotImplementedError()

def print_scores(cv):
    #print out cross validation scores
    [print('Crossvalidation fold: {}  Accruacy: {}'.format(n, score)) for n, score in enumerate(cv['test_score'])]
    #print out the mean of the cross validation
    print('Mean train cross validation score {}'.format(cv['test_score'].mean()))
    
print_scores(cv)


# In[ ]:


assert len(cv['test_score']) == 5
assert max(cv['test_score']) >= 0.85
assert min(cv['test_score']) >= 0.69
assert cv['test_score'].mean() >= 0.77


# ## Step 6: Creating an end to end ML pipeline
# Congraulations you've trained, crossvalidated, predicted, and evaluated your frist classifier. Now that you understand the basic steps we will look at a way to combine all these steps together.
# 
# Before we go further think about what you would have to do if you wanted to change the model. Intalize a new model, change the vairables, redo the cross validation...etc. Seems like a lot. And when we have to change lots of code it is easy to make mistakes. And what if you wanted to try many models and see which one performed best? Or try changing many different features? How could you do it without writing each one out as we have?
# 
# The solution is to use SKLearn's pipeline class. A pipeline is an object that will execute the various steps in the machine learning process. We can choose what elements we want in the pipeline and those that we do not. Once setup, we can rapidly change models, or input data and have it return our results in an ordered way.
# 
# 
# 1. Initalize a scaler and a classifer object like we did previously.
# 2. Use the ```make_pipeline()``` function to construct a transofmraiton pipeline for the scaler and the classifier
# 3. Input the pipeline object to the cross_validator and evaluate with 5 folds.
# 4. Print out your results (hint: make a function for repetitve tasks like printing)

# In[58]:


#define the scaler
#define the classifier
#make the pipeline
#run the cross validation
#print results
# scaler = 0
# classifier = 0
# pipe = 0
# scores = 0
# YOUR CODE HERE
#raise NotImplementedError()
scaler = preprocessing.StandardScaler()
classifier = LogisticRegression()
pipe = make_pipeline(scaler,classifier)
cv = cross_validate(pipe,X_train,Y_train,cv=5)
print_scores(cv)


# In[ ]:


assert type(pipe) == type(make_pipeline(scaler, classifier))
assert len(cv['test_score']) == 5
assert max(cv['test_score']) >= 0.83
assert min(cv['test_score']) >= 0.69
assert cv['test_score'].mean() >= 0.74


# ## Challenge Exercise
# 
# In this notebook we only used two features to predict the class of the flower. We also did not do any hypter parameter tuning. The challenge is to impove the prediction results. Some ideas we can try:
# 1. Add features to the input and run the cross validation pipeline
# 2. Investigate how to use ```GridSearchCV```, a powerful funtion that searches through hyperparmetrs and does cross validation.
#     - Hint: Input the pipeline directly into GridSearchCV
# 3. Try a different models like RandomForest or SVM.

# In[59]:


# YOUR CODE HERE
#raise NotImplementedError()
df = dataset_to_pandas()
X = df.drop(['target','class'], axis=1).to_numpy()
Y = df['target'].to_numpy()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)


# In[60]:


# YOUR CODE HERE
#raise NotImplementedError()
cv = cross_validate(pipe, X_train, Y_train, cv=5)
print_scores(cv)


# In[61]:


# YOUR CODE HERE
#raise NotImplementedError()
classifier = RandomForestClassifier()
cv = cross_validate(classifier, X_train, Y_train, cv=5)
print_scores(cv)


# In[62]:


# YOUR CODE HERE
#raise NotImplementedError()
scaler = preprocessing.StandardScaler()
classifier = SVC(C=10, kernel='linear')
pipe = make_pipeline(scaler, classifier)
cv = cross_validate(pipe, X_train, Y_train, cv=5)
print_scores(cv)


# In[ ]:


# YOUR CODE HERE
#raise NotImplementedError()

