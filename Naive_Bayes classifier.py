#!/usr/bin/env python
# coding: utf-8

# # Python Code for Implementing The Naive-Bayes Classifier 

# In[3]:


#importing the lbraries
import pandas as pd


# In[4]:


#loading the dataset
from sklearn.datasets import load_iris


# In[5]:


#displaying the dataset
iris=load_iris()
iris


# In[6]:


iris.feature_names#gives the independent attributes


# In[7]:


iris.target_names #gives the dependent attributes class labels


# In[8]:


X=iris.data #loading the independent values in the 'X' variable
X


# In[9]:


Y=iris.target #loading the independent values in the 'Y' variable
Y


# In[10]:


X.shape #checking the dimension of the 'X'


# In[11]:


Y.shape  #checking the dimension of the 'Y'


# # DATA SPLITTING INTO TRAIN AND TEST 

# In[12]:


#importing the train_test_split from sklearn in model selection
from sklearn.model_selection import train_test_split


# In[13]:


#the ratio of train and test is:(80:20)
#80% for training and 20% for testing 
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)


# In[14]:


X_train.shape #80% of 150 : num of attributes


# In[15]:


X_test.shape #20% of 150 :num of attributes


# In[16]:


Y_train.shape #80% of 150 :num of atatributes


# In[17]:


Y_test.shape #20% of 150 :num of atatributes


# # MODEL SELECTION

# In[18]:


#we are using naive bayes classification model 
from sklearn.naive_bayes import GaussianNB


# In[19]:


#method call 
model=GaussianNB()
#for training the model we use 'fit'keyword
#we use X_train and Y_train for training the model
model.fit(X_train,Y_train)


# # MODEL PREDICTION

# In[20]:


Y_pred=model.predict(X_test)
Y_pred


# # MODEL EVALUATION

# In[21]:


from sklearn.metrics import accuracy_score


# In[22]:


print("Navie Bayes Classifier accuracy on iris dataset is: ",accuracy_score(Y_pred,Y_test)*100)


# In[23]:


from sklearn.metrics import classification_report


# In[24]:


print("Classification report is \n",classification_report(Y_test,Y_pred))


# In[25]:


from sklearn.metrics import confusion_matrix


# In[26]:


print("Confusion matrix is: \n",confusion_matrix(Y_test,Y_pred))


# # DATA VISUALIZATION

# In[27]:


import matplotlib.pyplot as plt


# In[28]:


plt.bar(X[0],X[1],color="purple")
plt.title("BAR GRAPH OF IRIS DATASET")


# In[29]:


plt.pie(X[0])


# # IMPLEMENT THE SAME PROCEDURE FOR (70:30)RATIO TRAIN ANDTEST

# # DATA SPLITTING INTO TRAIN AND TEST

# In[30]:


from sklearn.model_selection import train_test_split


# In[32]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)


# In[33]:


X_train.shape


# In[34]:


Y_train.shape


# In[35]:


X_test.shape


# In[36]:


Y_test.shape


# # MODEL TRAINING

# In[37]:


from sklearn.naive_bayes import GaussianNB


# In[38]:


model=GaussianNB()
model


# In[39]:


model.fit(X_train,Y_train)


# # MODEL PREDICTION

# In[40]:


Y_pred=model.predict(X_test)
Y_pred


# # MODEL EVALUATION

# # CALCULATING THE SCORES(3)

# In[41]:


from sklearn.metrics import accuracy_score


# In[42]:


print("Naive Bayes classsifier accuracy score for iris dataset is: ",accuracy_score(Y_pred,Y_test)*100)


# In[43]:


from sklearn.metrics import classification_report


# In[44]:


print("Classification report is: \n",classification_report(Y_pred,Y_test))


# In[45]:


from sklearn.metrics import confusion_matrix


# In[46]:


print("confusion matrix is: \n",confusion_matrix(Y_pred,Y_test))


# In[47]:


import matplotlib.pyplot as plt


# In[48]:


plt.bar(X[0],X[1],color="yellow")
plt.title("SAMPLE BAR GRAPH")


# In[49]:


plt.pie(X[0])


# In[ ]:





# In[ ]:




