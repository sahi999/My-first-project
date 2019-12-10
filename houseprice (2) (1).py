#!/usr/bin/env python
# coding: utf-8

# ## Obtain Data

# ### Importing Required libraries

# In[2]:


import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# 
# 
# ### Reading Data From CSV File

# In[3]:


td=pd.read_csv("train.csv")


# ## Analyze The Data

# In[4]:


td.head()


# 
# ## Data Visualization

# In[5]:


td['LotShape'].value_counts()


# In[6]:


sb.heatmap(td.isnull(),yticklabels=False,cbar=False)


# In[7]:


td.shape


# In[8]:


td.info()
td.columns[td.isnull().any()]


# In[9]:


tdcorr = td.select_dtypes(include=[np.number])
tdcorr.shape


# #### Visualizing Top 50% Correlation Train
# #### Attributes With The SalePrice

# In[10]:


del tdcorr['Id']
corr = tdcorr.corr()
topattr = corr.index[abs(corr['SalePrice']>0.5)]
plt.subplots(figsize=(11, 8))
topcorr = td[topattr].corr()
sb.heatmap(topcorr, annot=True)
plt.show()


# In[11]:


print("the most important attributes relative to target")
corr = td.corr()
corr.sort_values(['SalePrice'], ascending=False, inplace=True)
corr.SalePrice


# ## Feature Engineering 
# #### Filling The missing Values In The Data

# In[12]:



td['PoolQC'] = td['PoolQC'].fillna('None')


# In[13]:


td['MiscFeature'] = td['MiscFeature'].fillna('None')
td['Alley'] = td['Alley'].fillna('None')
td['Fence'] = td['Fence'].fillna('None')
td['FireplaceQu'] = td['FireplaceQu'].fillna('None')


# In[14]:


td['LotFrontage'] = td.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))


# In[15]:


for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    td[col] = td[col].fillna('None')
#GarageYrBlt, GarageArea and GarageCars these are replacing with zero
for col in ['GarageYrBlt', 'GarageArea', 'GarageCars']:
    td[col] = td[col].fillna(int(0))
#BsmtFinType2, BsmtExposure, BsmtFinType1, BsmtCond, BsmtQual these are replacing with None
for col in ('BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual'):
    td[col] = td[col].fillna('None')
#MasVnrArea : replace with zero
td['MasVnrArea'] = td['MasVnrArea'].fillna(int(0))
#MasVnrType : replace with None
td['MasVnrType'] = td['MasVnrType'].fillna('None')
#There is put mode value 
td['Electrical'] = td['Electrical'].fillna(td['Electrical']).mode()[0]
#There is no need of Utilities
td = td.drop(['Utilities'], axis=1)


# In[16]:


plt.figure(figsize=(10, 5))
sb.heatmap(td.isnull())


# In[17]:


colmn = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold', 'MSZoning', 'LandContour', 'LotConfig', 'Neighborhood',
        'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
        'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'Foundation', 'GarageType', 'MiscFeature', 
        'SaleType', 'SaleCondition', 'Electrical', 'Heating')
len(colmn)


# ### Encoding String To Integer Using LabelEncoder

# In[18]:


from sklearn.preprocessing import LabelEncoder
for i in colmn:
    lb = LabelEncoder() 
    lb.fit(list(td[i].values)) 
    td[i] = lb.transform(list(td[i].values))


# In[19]:


#Take targate variable into y
y = td['SalePrice']
#Delete the saleprice
del td['SalePrice']
#Take their values in X and y
X = td.values
y = y.values


# 
# ## Train-Test Split

# In[20]:


# Split data into train and test formate
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)


# ### Training The Model

# In[21]:


#Train the model
from sklearn import linear_model
model = linear_model.LinearRegression()
#Fit the model
model.fit(X_train, y_train)


# In[22]:


print("Predict value " + str(model.predict([X_test[131]])))
print("Real value " + str(y_test[131]))


# In[23]:


print("Accuracy --> ", model.score(X_test, y_test)*100)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=1000)
#Fit
model.fit(X_train, y_train)


# In[ ]:


print("Accuracy --> ", model.score(X_test, y_test)*100)


# ## Predicting The Model Using-
# #### => Linear Regression
# #### => Random Forest Regression
# ####  =>Gradient Boosting
# 

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
gradboost = GradientBoostingRegressor(n_estimators=100, max_depth=4)
#Fit
gradboost.fit(X_train, y_train)


# In[ ]:


print("Accuracy --> ", gradboost.score(X_test, y_test)*100)


# In[ ]:


pred=gradboost.predict(X)


# In[ ]:


pred
len(pred)


# ### Submitting The Predicted Result

# In[ ]:


predi=pd.DataFrame(pred[:-1])
sd=pd.read_csv('sample_submission.csv')
submit=pd.concat([sd['Id'],predi],axis=1)
submit.columns=['Id','SalePrice']
submit.to_csv('result1.csv',index=False)


# In[ ]:





# In[ ]:




