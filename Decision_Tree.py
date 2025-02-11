#!/usr/bin/env python
# coding: utf-8

# # Python Code for Implementing The Decision-Tree

# In[1]:


import pandas as pd


# In[2]:


from sklearn.datasets import load_iris


# In[3]:


iris=load_iris()
iris


# In[4]:


iris.feature_names


# In[5]:


iris.target_names


# In[6]:


iris.data


# In[7]:


iris.target


# In[8]:


X=iris.data
X


# In[9]:


Y=iris.target
Y


# In[10]:


X.shape


# In[11]:


Y.shape


# # DATA SPLITTING INTO TRAIN AND TEST

# In[12]:


from sklearn.model_selection import train_test_split 


# In[13]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)


# In[14]:


X_train.shape


# In[15]:


X_test.shape


# In[16]:


Y_train.shape


# In[17]:


Y_test.shape


# # MODEL SELECTION

# In[18]:


from sklearn.tree import DecisionTreeClassifier


# In[19]:


model=DecisionTreeClassifier()
model.fit(X_train,Y_train)


# # MODEL PREDICTION

# In[20]:


Y_pred=model.predict(X_test)
Y_pred


# # MODEL EVALUATION

# In[21]:


from sklearn.metrics import accuracy_score


# In[22]:


print("Accuracy score of iris data set is ",accuracy_score(Y_pred,Y_test)*100)


# In[23]:


from sklearn.metrics import classification_report


# In[24]:


print("Classification report is \n",classification_report(Y_pred,Y_test))


# In[25]:


from sklearn.metrics import confusion_matrix


# In[34]:


print("Confusion matrix is \n",confusion_matrix(Y_pred,Y_test))


# # TESTING ON NEW DATA..

# In[27]:


import numpy as np


# In[28]:


new_input=np.array([[10,2,3,4]])


# In[29]:


model.predict(new_input)


# In[30]:


label=model.predict(new_input)


# In[31]:


if(label[0]==0):
    print("setosa")
elif(label[0]==1):
    print("versicolor")
else:
    print("virginica")


# In[32]:


a=np.array([[5.1, 3.5, 1.4, 0.2]])
b=model.predict(a)
model.predict(a)


# In[33]:


if(b[0]==0):
    print("setosa")
elif(b[0]==1):
    print("versicolor")
else:
    print("virginica")

