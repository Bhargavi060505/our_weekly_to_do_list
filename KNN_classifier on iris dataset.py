#!/usr/bin/env python
# coding: utf-8

# # Implement KNN algorithm for Iris dataset

# In[1]:


#import the libraries
import pandas as pd


# In[2]:


from sklearn.datasets import load_iris


# In[3]:


iris_data=load_iris()
iris_data


# In[4]:


iris_data.feature_names # independent attribute names


# In[5]:


iris_data.target_names #dependent variable's class labels 


# # Separating into X and Y variables

# In[6]:


X=iris_data.data
X


# In[7]:


X.shape


# In[8]:


Y=iris_data.target
Y


# In[9]:


Y.shape


# # splitting into train and test ratio..(80:20)

# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)


# In[12]:


X_train.shape


# In[13]:


X_test.shape


# In[14]:


Y_train.shape


# In[15]:


Y_test.shape


# # Model Selection-KNN Classifier

# In[16]:


from sklearn.neighbors import KNeighborsClassifier


# In[17]:


model=KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,Y_train)


# # Model Prediction

# In[18]:


Y_pred=model.predict(X_test)
Y_pred


# In[19]:


Y_pred.shape


# # Model Evaluation

# In[20]:


from sklearn.metrics import accuracy_score


# In[21]:


print("Accuracy of the KNN classifier is: ",accuracy_score(Y_test,Y_pred)*100)


# In[22]:


from sklearn.metrics import classification_report


# In[23]:


print("Classification report: \n",classification_report(Y_test,Y_pred))


# In[24]:


from sklearn.metrics import confusion_matrix


# In[25]:


print("Confusion Matrix is: \n",confusion_matrix(Y_pred,Y_test))


# # Testing on New Instance

# In[26]:


import numpy as np


# In[27]:


new_input=np.array([[1,2,3,4]])


# In[28]:


label=model.predict(new_input)


# In[29]:


label


# In[31]:


if(label[0]==0):
    print("Setosa")
elif(label[0]==1):
    print("Versicolor")
else:
    print("Virginica")


# In[ ]:




