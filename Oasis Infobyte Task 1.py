#!/usr/bin/env python
# coding: utf-8

# ## Oasis Infobyte 
# ### Author : Kailas Rayappa Gadade
# ### Task 1 : PERFOMING MACHINE LEARNING MODEL ON IRIS FLOWER DATASET

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score


# In[2]:


data=pd.read_csv("C:\\Users\\Kailas\\OneDrive\\Desktop\\Iris.csv")
data.head()


# In[3]:


data.tail() 


# In[4]:


data.isnull().sum() 


# In[5]:


data.shape 


# In[6]:


data.dtypes 


# In[7]:


data['Species'].unique() 


# In[8]:


data.describe() 


# # Data Visualization

# In[9]:


sns.pairplot(data)


# In[10]:


data.corr()


# In[11]:


sns.heatmap(data.corr(),annot=True)
plt.show()


# In[12]:


plt.boxplot(data['SepalLengthCm'])
plt.show()


#  From above heatmap we can see that there is no outliear in the SepalLengthCm

# In[13]:


plt.boxplot(data['SepalWidthCm'])
plt.show()


# from above boxplot we can see that there are some outliear predict in SepalWidth

# In[14]:


plt.boxplot(data['PetalLengthCm'])
plt.show()


# from above boxplot we can see that there are some outliear predict in PetalLength

# In[15]:


plt.boxplot(data['PetalWidthCm'])
plt.show()


# from above boxplot we can see that there are some outliear predict in PetalWidth

# In[16]:


data.drop('Id',axis=1, inplace=True)


# In[17]:


spec={'Iris-setosa':1,'Iris-versicolor':2, 'Iris-virginica':3}
data.Species=[spec[i] for i in data.Species]
data


# In[18]:


x=data.iloc[:,0:4]
x


# In[19]:


y=data.iloc[:,4]
y


# In[20]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)


# ## Training Model

# In[21]:


model=LinearRegression()


# In[22]:


model.fit(x,y)


# In[23]:


model.score(x,y)  


# In[24]:


model.coef_


# In[25]:


model.intercept_


# ## Making Predictions

# In[26]:


y_pred=model.predict(x_test)


# ## Model Evolation

# In[27]:


print("Mean Squared Error: %.2f" % np.mean((y_pred - y_test)**2))


# # Naive Bayes Algorithm

# In[28]:


from sklearn.naive_bayes import GaussianNB
accuracies={}
nb = GaussianNB()
nb.fit(x_train, y_train)
acc=nb.score(x_test,y_test)*100
accuracies['Naive Bayes']=acc
print("Accuracy Of Naive Bayes:{:.2f}%".format(acc))


# In[29]:


nb.score(x_train,y_train)*100


# In[30]:


from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
predictions = nb.predict(x_test)
predictions
sns.heatmap(confusion_matrix(y_test, predictions), annot = True)
plt.show()


# ### Thank You...

# In[ ]:




