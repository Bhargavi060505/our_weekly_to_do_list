#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


heart=pd.read_csv('heart.csv')


# In[3]:


heart


# In[4]:


heart.columns


# In[5]:


Y=heart['target']
Y


# In[6]:


X=heart.drop(['target'],axis=1)
X


# In[7]:


X.shape


# In[8]:


Y.shape


# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)


# In[11]:


X_train.shape 


# In[12]:


X_test.shape


# In[13]:


Y_train.shape


# In[14]:


Y_test.shape


# In[15]:


#model selection-knn classifier


# In[16]:


from sklearn.neighbors import KNeighborsClassifier


# In[17]:


model=KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,Y_train)


# In[18]:


#model prediction


# In[19]:


Y_pred=model.predict(X_test)
Y_pred


# In[20]:


Y_pred.shape


# In[21]:


#model evaluation


# In[22]:


from sklearn.metrics import accuracy_score


# In[23]:


print("accuracy score of knn classifier is: ",accuracy_score(Y_test,Y_pred)*100)


# In[24]:


from sklearn.metrics import classification_report


# In[25]:


print("Classification report is: \n",classification_report(Y_pred,Y_test))


# In[26]:


from sklearn.metrics import confusion_matrix


# In[27]:


print("confusion matrix is: \n",confusion_matrix(Y_pred,Y_test))


# In[ ]:




