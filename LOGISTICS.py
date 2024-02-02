#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
logdf=pd.read_csv('insurance_data.csv')
lindf=pd.read_csv('insurance_data.csv')


# In[6]:


logdf


# In[8]:


from matplotlib import pyplot as plt
plt.scatter(lindf['age'],lindf['bought_insurance'],marker='+',color='green')  # scatter IS USED TO MARK THE POINTS IN THE SCATTERED MANNER.


# In[12]:


from sklearn.linear_model import LogisticRegression
x=logdf[['age']]
y=logdf[['bought_insurance']]


# In[13]:


model=LogisticRegression()


# In[14]:


model.fit(x,y)


# In[15]:


logan=model.predict([[18],[22],[47]])


# In[16]:


logan


# In[19]:


model.score(x,y)


# In[21]:


from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(logdf[['age']],logdf[['bought_insurance']],test_size=0.15)


# In[22]:


modtt=LogisticRegression()
modtt.fit(x_train,y_train)


# In[24]:


ttans=modtt.predict(x_test)
ttans


# In[25]:


modtt.score(x_train,y_train)


# In[27]:


model.predict_proba([[18],[22],[43]])


# In[28]:


modtt.predict_proba([[18],[22],[43]])


# In[55]:


logdf1=pd.read_csv('HR_comma_sep.csv')
lindf1=pd.read_csv('HR_comma_sep.csv')
logdf1


# In[30]:


print(logdf1.to_string())


# In[31]:


left=logdf1[logdf1['left']==1]
left.shape


# In[32]:


retain=logdf1[logdf1['left']==0]
retain.shape


# In[34]:


nlogdf=logdf1.drop(columns=['Department','salary'])
nlogdf.groupby('left').mean()


# In[39]:


pd.crosstab(logdf1['salary'],logdf1['left']).plot(kind='line')


# In[40]:


pd.crosstab(logdf1['salary'],logdf1['left']).plot(kind='bar')


# In[45]:


pd.crosstab(logdf1['Department'],logdf1['left']).plot(kind='bar')


# In[56]:


lindf1=lindf1.drop(columns=['Department'])
lindf1=pd.get_dummies(lindf1,dtype=int)
print(lindf1.to_string())


# In[57]:


lindf1


# In[68]:


x2=lindf1.drop(columns=['last_evaluation','number_project','time_spend_company','Work_accident','left'])
y2=lindf1['left']
print(x2.to_string())


# In[69]:


from sklearn.linear_model import LogisticRegression as lor


# In[70]:


mod4=LogisticRegression()


# In[71]:


mod4.fit(x2,y2)


# In[75]:


ans=mod4.predict([[0.89,224,0,0,1,0]])
print(ans)


# In[76]:


ans.score


# In[77]:


mod4.score(x2,y2)


# In[ ]:




