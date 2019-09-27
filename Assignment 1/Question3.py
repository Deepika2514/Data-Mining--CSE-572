#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv(r"C:\Users\Deepika\Downloads\overdoses.csv")


# In[3]:


print(df)


# In[4]:


df["Population"]=df["Population"].str.replace(",","")


# In[5]:


df


# In[6]:


df["Deaths"]=df["Deaths"].str.replace(",","")


# In[7]:


df


# In[8]:


df["Population"]=df["Population"].apply(int)


# In[9]:


df["Deaths"]=df["Deaths"].apply(int)


# In[10]:


df1={"ODD":df.Deaths/df.Population,"State code":df.Abbrev}


# In[11]:


print(df1)


# In[12]:


df1


# In[13]:


similarity_matrix=np.zeros[(df1.Abbrev.shape(1),df1.Abbrev.shape(1))]


# In[14]:


df1=pd.dataframe(df1)


# In[ ]:


df1=pd.Dataframe(df1)


# In[16]:


df1=pd.DataFrame(df1)


# In[20]:


similarity_matrix=np.zeros[(df1.shape[1],df1.shape[1])]


# In[21]:


df1.shape[1]


# In[22]:


df1.shape[2]


# In[23]:


df1


# In[24]:


len(df1.index)


# In[25]:


similarity_matrix=np.zeros[((len(df1.index)+1),(len(df1.index)+1))]


# In[26]:


df1.to_numpy()


# In[27]:


df1.dtypes


# In[28]:


df1.dtypes


# In[30]:


arr_req=df1.values


# In[31]:


print(arr_req)


# In[32]:


arr_req


# In[34]:


len(arr_req(index))


# In[35]:


arr_req.shape


# In[38]:


arr_req.shape[0]


# In[40]:


similarity_matrix=np.zeros((((arr_req.shape[0])+1),((arr_req.shape[0])+1)))


# In[41]:


similarity_matrix


# In[42]:


similarity_matrix.shape


# In[47]:


similarity_matrix[:,1]=arr_req[:,1].copy()


# In[46]:





# In[ ]:




