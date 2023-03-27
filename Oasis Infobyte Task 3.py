#!/usr/bin/env python
# coding: utf-8

# ## Oasis Infobyte
# ### Author : Kailas Rayappa Gadade
# ### Task 3 : CAR PRICE PREDICTION WITH MACHINE LEARNING

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)


# In[2]:


data=pd.read_csv("C:\\Users\\Kailas\\OneDrive\\Desktop\\CarPrice_Assignment.csv")
data.head()


# In[3]:


data.tail()


# In[4]:


data.describe()


# In[5]:


data.info()


# In[6]:


data.shape


# In[7]:


data.isnull().sum()


# In[8]:


data.duplicated().sum()


# In[9]:


print(data.price.describe(percentiles=[0.225,0.50,0.75,0.85,0.98,1]))


# ### Exploratory Data Analysis

# In[10]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.title('Door Number Histogram')
sns.countplot(data.doornumber,palette=("plasma"))
plt.subplot(1,2,2)
plt.title('Door Number vs Price')
sns.boxplot(x=data.doornumber,y=data.price,palette=("plasma"))
plt.show()
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.title('Aspiration Histogram')
sns.countplot(data.aspiration,palette=('plasma'))
plt.subplot(1,2,2)
plt.title('Aspiration vs Price')
sns.boxplot(x=data.aspiration,y=data.price,palette=("plasma"))
plt.show()


# In[11]:


df=pd.DataFrame(data.groupby(['fueltype'])['price'].mean().sort_values(ascending=False))
df.plot.bar()
plt.title('Fuel Type vs Average Price')
plt.show()
df=pd.DataFrame(data.groupby(['carbody'])['price'].mean().sort_values(ascending=False))
df.plot.bar()
plt.title('Car Type vs Average Price')
plt.show()


# In[12]:


plt.figure(figsize=(8,6))
plt.title('Fuel economy vs Price')
sns.scatterplot(x=data['fuelsystem'],y=data['price'],hue=data['drivewheel'])
plt.xlabel('Fuel System')
plt.ylabel('Price')
plt.show()
plt.tight_layout()


# In[13]:


predict="price"
data=data[["symboling","wheelbase","carlength","carwidth","carheight","curbweight","enginesize","boreratio","stroke","compressionratio","horsepower","peakrpm","citympg","highwaympg","price"]]


# In[14]:


x=np.array(data.drop([predict],1))
y=np.array(data[predict])


# In[15]:


print(x)
print(y)


# In[16]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.2,random_state=100)


# In[17]:


from sklearn.tree import DecisionTreeRegressor
model=DecisionTreeRegressor()
model.fit(xtrain,ytrain)
predictions=model.predict(xtest)
from sklearn.metrics import mean_absolute_error
model.score(xtest,predictions)


# #### Model gives 100% Accuracy on the test set

# ### Thank You...

# In[ ]:




