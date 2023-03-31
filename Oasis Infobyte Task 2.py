#!/usr/bin/env python
# coding: utf-8

# ## Oasis Infobyte 
# ### Author : Kailas Rayappa Gadade
# ### Task 2 : PERFOMING EDA ON UMEMPLOYMENT RATE IN INDIA

# ### Step1:Import libarires

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import calendar


# In[2]:


import datetime as dt

import plotly.io as pio
pio.templates


# In[3]:


import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from IPython.display import HTML


# ## Step 2: 

# In[4]:


df = pd.read_csv("C:\\Users\\Kailas\\OneDrive\\Desktop\\Unemployment_Rate_upto_11_2020.csv")


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


df.columns =['States','Date','Frequency','Estimated Unemployment Rate','Estimated Employed','Estimated Labour Participation Rate','Region','longitude','latitude']


# In[9]:


df['Date'] = pd.to_datetime(df['Date'],dayfirst=True)
df['Frequency']= df['Frequency'].astype('category')
df['Month'] =  df['Date'].dt.month
df['Month_int'] = df['Month'].apply(lambda x : int(x))
df['Month_name'] =  df['Month_int'].apply(lambda x: calendar.month_abbr[x])
df['Region'] = df['Region'].astype('category')
df.drop(columns='Month',inplace=True)
df.head(3)


# In[10]:


df_stats = df[['Estimated Unemployment Rate',
       'Estimated Employed', 'Estimated Labour Participation Rate']]


round(df_stats.describe().T,2)


# In[11]:


region_stats = df.groupby(['Region'])[['Estimated Unemployment Rate','Estimated Employed','Estimated Labour Participation Rate']].mean().reset_index()

region_stats = round(region_stats,2)


region_stats


# In[12]:


heat_maps = df[['Estimated Unemployment Rate',
       'Estimated Employed', 'Estimated Labour Participation Rate',
       'longitude', 'latitude', 'Month_int']]

heat_maps = heat_maps.corr()

plt.figure(figsize=(10,6))
sns.set_context('notebook',font_scale=1)
sns.heatmap(heat_maps, annot=True,cmap='ocean');


# In[13]:


fig = px.box(df,x='States',y='Estimated Unemployment Rate',color='States',title='Unemployment rate',template='plotly')
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# In[14]:


fig = px.bar(df, x='Region',y='Estimated Unemployment Rate',animation_frame = 'Month_name',color='States',
            title='Unemployment rate across region from Jan.2020 to Oct.2020', height=700,template='plotly')

fig.update_layout(xaxis={'categoryorder':'total descending'})

fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 2000

fig.show()


# In[15]:


unemplo_df = df[['States','Region','Estimated Unemployment Rate','Estimated Employed','Estimated Labour Participation Rate']]

unemplo = unemplo_df.groupby(['Region','States'])['Estimated Unemployment Rate'].mean().reset_index()
fig = px.sunburst(unemplo, path=['Region','States'], values='Estimated Unemployment Rate',
                  color_continuous_scale='Plasma',title= 'unemployment rate in each region and state',
                  height=650,template='ggplot2')


fig.show()


# In[16]:


fig = px.scatter_geo(df,'longitude', 'latitude', color="Region",
                     hover_name="States", size="Estimated Unemployment Rate",
                     animation_frame="Month_name",scope='asia',template='plotly',title='Impack of lockdown on employement across regions')

fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 2000

fig.update_geos(lataxis_range=[5,35], lonaxis_range=[65, 100],oceancolor="blue",
    showocean=True)

fig.show()


# In[17]:


data = pd.read_csv("C:\\Users\\Kailas\\OneDrive\\Desktop\\Unemployment in India.csv")
data.head()


# In[18]:


data.tail()


# In[19]:


data.info()


# In[20]:


data.isnull().sum()


# In[21]:


data.describe()


# In[22]:


data.corr()


# In[23]:


sns.heatmap(data.corr(),annot = True)


# In[24]:


sns.pairplot(data, hue="Region");


# In[25]:


freq = data['Region'].value_counts()
freq


# In[26]:


freq.plot(kind='pie',startangle = 90)
plt.legend()
plt.show()


# In[27]:


sns.countplot(x='Region',data=data)
plt.xticks(rotation=45)
plt.ylabel('Estimated Unemployment Rate (%)')


# In[28]:


data.hist()
plt.show()


# ### Thank You...
