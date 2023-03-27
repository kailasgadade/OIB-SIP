#!/usr/bin/env python
# coding: utf-8

# ## Oasis Infobyte
# ### Author : Kailas Rayappa Gadade
# ### Task 5 : SALES PREDICTION USING PYTHON

# ### Importing libraries

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ### Importing Dataset

# In[2]:


data=pd.read_csv("C:\\Users\\Kailas\\OneDrive\\Desktop\\Advertising.csv")


# In[3]:


data.sample(5)


# In[4]:


data.tail()


# In[5]:


data.shape


# In[6]:


data.describe()


# In[7]:


data.isnull().sum()


# In[8]:


data.info()


# ### Data Visualisation

# In[9]:


data.corr()


# In[10]:


sns.heatmap(data.corr(),cbar=True,linewidths=0.5,annot=True)


# In[11]:


sns.pairplot(data)


# In[12]:


sns.distplot(data['Newspaper'])


# In[13]:


sns.distplot(data['Radio'])


# In[14]:


sns.distplot(data['Sales'])


# In[15]:


sns.distplot(data['TV'])


# ### Data Preprosasing

# In[16]:


data=data.drop(columns=['Unnamed: 0'])


# In[17]:


data


# In[18]:


x=data.drop(['Sales'],1)
x.head()


# In[19]:


y=data['Sales']


# In[20]:


y.head()


# ### Spliting the Dataset

# In[21]:


x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.3, random_state=42)


# In[22]:


print(x.shape,x_train.shape,x_test.shape)


# In[23]:


print(y.shape,y_train.shape,y_test.shape)


# In[24]:


x_train=x_train.astype(int)
y_train=y_train.astype(int)
x_test=x_test.astype(int)
y_test=y_test.astype(int)


# In[25]:


from sklearn.preprocessing import StandardScaler
Sc=StandardScaler()
x_train_scaled=Sc.fit_transform(x_train)
x_test_scaled=Sc.fit_transform(x_test)


# ### Applying Linear Regression

# In[26]:


from sklearn.linear_model import LinearRegression
accuracies={}
lr=LinearRegression()
lr.fit(x_train,y_train)
acc=lr.score(x_test,y_test)*100
accuracies['Linear Regression']=acc
print("Test Accuracy {:.2f}%".format(acc))


# ### Analyzing the data by Scatter plot

# In[27]:


y_pred=lr.predict(x_test_scaled)


# In[28]:


plt.scatter(y_test,y_pred,c='r')


# ### Thank You...

# In[ ]:




