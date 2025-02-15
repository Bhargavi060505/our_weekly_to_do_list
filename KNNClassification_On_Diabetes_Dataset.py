#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


diabetes=pd.read_csv('diabetes.csv')


# In[3]:


diabetes


# In[6]:


diabetes.columns


# In[7]:


Y=diabetes['Outcome']
Y


# In[8]:


X=diabetes.drop(['Outcome'],axis=1)
X


# In[10]:


X.shape


# In[11]:


Y.shape


# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)


# In[16]:


X_train.shape #768 of 80%


# In[17]:


X_test.shape


# In[18]:


Y_train.shape


# In[19]:


Y_test.shape


# In[20]:


#model selection-knn classifier


# In[21]:


from sklearn.neighbors import KNeighborsClassifier


# In[25]:


model=KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,Y_train)


# In[26]:


#model prediction


# In[27]:


Y_pred=model.predict(X_test)
Y_pred


# In[30]:


Y_pred.shape


# In[28]:


#model evaluation


# In[31]:


from sklearn.metrics import accuracy_score


# In[32]:


print("accuracy score of knn classifier is: ",accuracy_score(Y_test,Y_pred)*100)


# In[36]:


from sklearn.metrics import classification_report


# In[37]:


print("Classification report is: \n",classification_report(Y_pred,Y_test))


# In[33]:


from sklearn.metrics import confusion_matrix


# In[35]:


print("confusion matrix is: \n",confusion_matrix(Y_pred,Y_test))


# In[ ]:




