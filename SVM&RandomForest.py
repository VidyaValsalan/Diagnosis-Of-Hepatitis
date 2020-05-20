#!/usr/bin/env python
# coding: utf-8

# #  Import 

# In[18]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Load Hepatitis Data

# In[19]:

cell_df = pd.read_csv('C:\datamining\hepatitis.csv',na_values = '?',names=['Class','AGE','SEX','STEROID','ANTIVIRALS','FATIGUE','MALAISE','ANOREXIA','LIVER BIG','LIVER FIRM','SPLEEN PALPABLE','SPIDERS','ASCITES','VARICES','BILIRUBIN','ALK PHOSPHATE','SGOT','ALBUMIN','PROTIME','HISTOLOGY'])

# # Data Cleaning

# In[20]:

#Plotshowing the missing value
sns.heatmap(cell_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')

# In[21]:

#Ignore the attribute PROMITE contains several tuples with missing values.
del cell_df['PROTIME']
#Ignore row 56 contains several attributes with missing values.
cell_df.drop([56],axis=0,inplace=True)

# # Use a measure of central tendency for the attribute (e.g., the mean or median) to fill in the missing value

# In[22]:

# For normal (symmetric) data distributions, the mean can be used, while skewed data distribution should employ the median
# Graph showing the attribute ALK PHOSPHATE is left skewed
sns.distplot(cell_df['ALK PHOSPHATE'].dropna(),kde=False,color='darkred',bins=40)

# In[23]:

#Replace with median
cell_df['ALK PHOSPHATE'].fillna((cell_df['ALK PHOSPHATE'].median()), inplace=True)

# In[24]:

# Graph showing the attribute ALBUMIN is normally distributed
sns.distplot(cell_df['ALBUMIN'].dropna(),kde=False,color='darkred',bins=40)

# In[25]:

#missing value of ALBUMIN replace with mean value
cell_df['ALBUMIN'].fillna((cell_df['ALBUMIN'].mean()), inplace=True)
#steroid replace with median
cell_df['STEROID'].fillna((cell_df['STEROID'].median()), inplace=True)
#replace liverbig with median
cell_df['LIVER BIG'].fillna((cell_df['LIVER BIG'].median()), inplace=True)
#replace liverfirm with median
cell_df['LIVER FIRM'].fillna((cell_df['LIVER FIRM'].median()), inplace=True)
#replace SPLEEN PALPABLE with median
cell_df['SPLEEN PALPABLE'].fillna((cell_df['SPLEEN PALPABLE'].median()), inplace=True)
#replace SPIDERS with median
cell_df['SPIDERS'].fillna((cell_df['SPIDERS'].median()), inplace=True)
#replace ASCITES with median
cell_df['ASCITES'].fillna((cell_df['ASCITES'].median()), inplace=True)
#replace VARICES with median
cell_df['VARICES'].fillna((cell_df['VARICES'].median()), inplace=True)
#replace BILIRUBIN with median
cell_df['BILIRUBIN'].fillna((cell_df['BILIRUBIN'].median()), inplace=True)
#replace SGOT with median
cell_df['SGOT'].fillna((cell_df['SGOT'].median()), inplace=True)

# In[26]:

#Plot showing there is no missing value
sns.heatmap(cell_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[27]:
#Check all the types are numerical or not
cell_df.dtypes

# In[28]:
ft_df=cell_df[['AGE','SEX','STEROID','ANTIVIRALS','FATIGUE','MALAISE','ANOREXIA','LIVER BIG','LIVER FIRM','SPLEEN PALPABLE','SPIDERS','ASCITES','VARICES','BILIRUBIN','ALK PHOSPHATE','SGOT','ALBUMIN','HISTOLOGY']]
x=np.asarray(ft_df)
y=np.asarray(cell_df['Class'])

# In[29]:
# For testing purpose, divide the data into the 80/20 training set and testing set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)

# # SVM Model

# In[30]:
from sklearn import svm
classifier=svm.SVC(kernel='linear',gamma='auto',C=2)
classifier.fit(x_train,y_train)
y_predict=classifier.predict(x_test)
#y_predict

# In[31]:
#Checking the result
from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict))


# # Random Forest Model

# In[32]:
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=100)
rf.fit(x_train,y_train)
y1_predict=rf.predict(x_test)


# In[15]:
#Checking the result
from sklearn.metrics import classification_report
print(classification_report(y_test,y1_predict))
