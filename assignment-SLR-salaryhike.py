# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 18:58:36 2023

@author: user
"""

import numpy as np
import pandas as pd

df=pd.read_csv("D:\\Data science\\Salary_Data.csv")
df
df1=df.rename({'Salary':'salary','YearsExperience':'Years_Experience'},axis=1)
df1
df.describe()
df.info()
df[df.values==0.0]
df.isnull().sum()
df[df.duplicated()].shape
df[df.duplicated()]

#EDA
import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(df['YearsExperience'])
plt.show()
sns.boxplot(df['Salary'])
plt.show()
#scatter plot
plt.scatter(df['YearsExperience'],df['Salary'])
plt.show()
#histogram
df['YearsExperience'].hist()
df['Salary'].hist()
#calculating correlation
df.corr()

import matplotlib.pyplot as plt
plt.scatter(df['Salary'],df['YearsExperience'])
plt.title('Homocadasticity')
plt.show()
#histogram

#for salary applying transformaton
sns.displot(df['Salary'],bins=6,kde=True)
plt.title('before transformation')
sns.displot(np.log(df['Salary']))
plt.title('after transformation')

import statsmodels.formula.api as smf
import statsmodels.api as smf
smf.qqplot(df['Salary'],line='r')
plt.title('before transformation')
smf.qqplot(np.log(df['Salary']),line='r')
plt.title('after log transformation')
smf.qqplot(np.sqrt(df['Salary']),line='r')
plt.title('after squareroot transformation')
smf.qqplot(np.cbrt(df['Salary']),line='r')
plt.title('after cuberoot transformation')
plt.show()

#for yers experience

sns.displot(df['YearsExperience'],bins=6,kde=True)
plt.title('before transformation')
sns.displot(np.log(df['YearsExperience']))
plt.title('after transformation')

import statsmodels.api as smf
smf.qqplot(df['YearsExperience'],line='r')
plt.title('before transformation')
smf.qqplot(np.log(df['YearsExperience']),line='r')
plt.title('after log transformation')
smf.qqplot(np.sqrt(df['YearsExperience']),line='r')
plt.title('after squareroot transformation')
smf.qqplot(np.cbrt(df['YearsExperience']),line='r')
plt.title('after cuberoot transformation')
plt.show()

#model fitting
import statsmodels.formula.api as sm
model=sm.ols('salary~Years_Experience', data = df1).fit()
model.summary()
square_root=sm.ols('np.sqrt(salary)~np.sqrt(Years_Experience)',data=df1).fit()
square_root.summary()
cube_model=sm.ols('np.cbrt(salary)~np.cbrt(Years_Experience)',data=df1).fit()
cube_model.summary()
log_model=sm.ols('np.log(salary)~np.log(Years_Experience)',data=df1).fit()
log_model.summary()
model.params
print(model.tvalues,'/n',model.pvalues)
predicted2=pd.DataFrame()
predicted2['Years_Experience']=df1.Years_Experience
predicted2['salary']=df1.salary
predicted2['predicted_delivery_time']=pd.DataFrame(model.predict(predicted1.Years_Experience))
predicted2
