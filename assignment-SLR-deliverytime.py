# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 18:58:00 2023

@author: user
"""

import numpy as np
import pandas as pd

df=pd.read_csv("D:\\Data science\\delivery_time.csv")
df
df1=df.rename({'Delivery Time':'Delivery_Time','Sorting Time':'Sorting_Time'},axis=1)
df1
df.describe()
df.info()
df[df.duplicated()].shape
df[df.duplicated()]
df=df.drop_duplicates().reset_index(drop=True)
df[df.duplicated()]


df[df.values==0.0]
df.isnull().sum()
#boxplot
import matplotlib.pyplot as plt
import seaborn as sns
sns.boxplot(df['Delivery Time'])
plt.title('delivery time')
plt.show()

sns.boxplot(df['Sorting Time'])
plt.title('sorting time')
plt.show()
#calculating correlation
df.corr()


import matplotlib.pyplot as plt
plt.scatter(df['Delivery Time'],df['Sorting Time'])
plt.title('Heteroscadasticity')
plt.show()
#histogram
df['Delivery Time'].hist()
df['Sorting Time'].hist()

#for delivery time applying transformaton
sns.displot(df['Delivery Time'],bins=6,kde=True)
plt.title('before transformation')
sns.displot(np.log(df['Delivery Time']))
plt.title('after transformation')

import statsmodels.formula.api as smf
import statsmodels.api as smf
smf.qqplot(df['Delivery Time'],line='r')
plt.title('before transformation')
smf.qqplot(np.log(df['Delivery Time']),line='r')
plt.title('after log transformation')
smf.qqplot(np.sqrt(df['Delivery Time']),line='r')
plt.title('after squareroot transformation')
smf.qqplot(np.cbrt(df['Delivery Time']),line='r')
plt.title('after cuberoot transformation')
plt.show()

#for sorting time

sns.displot(df['Sorting Time'],bins=6,kde=True)
plt.title('before transformation')
sns.displot(np.log(df['Sorting Time']))
plt.title('after transformation')

import statsmodels.api as smf
smf.qqplot(df['Sorting Time'],line='r')
plt.title('before transformation')
smf.qqplot(np.log(df['Sorting Time']),line='r')
plt.title('after log transformation')
smf.qqplot(np.sqrt(df['Sorting Time']),line='r')
plt.title('after squareroot transformation')
smf.qqplot(np.cbrt(df['Sorting Time']),line='r')
plt.title('after cuberoot transformation')
plt.show()


#model fitting
import statsmodels.formula.api as sm
model=sm.ols('Delivery_Time~Sorting_Time', data = df1).fit()
model.summary()
square_root=sm.ols('np.sqrt(Delivery_Time)~np.sqrt(Sorting_Time)',data=df1).fit()
square_root.summary()
cube_model=sm.ols('np.cbrt(Delivery_Time)~np.cbrt(Sorting_Time)',data=df1).fit()
cube_model.summary()
log_model=sm.ols('np.log(Delivery_Time)~np.log(Sorting_Time)',data=df1).fit()
log_model.summary()
model.params
print(model.tvalues,'/n',model.pvalues)
#with log transformation
predicted=pd.DataFrame()
predicted['Sorting_Time']=df1.Sorting_Time
predicted['Delivery_Time']=df1.Delivery_Time
predicted['predicted_delivery_time']=pd.DataFrame(np.exp(log_model.predict(predicted)))
predicted
#with original model
predicted1=pd.DataFrame()
predicted1['Sorting_Time']=df1.Sorting_Time
predicted1['Delivery_Time']=df1.Delivery_Time
predicted1['predicted_delivery_time']=pd.DataFrame(model.predict(predicted1.Sorting_Time))
predicted1
