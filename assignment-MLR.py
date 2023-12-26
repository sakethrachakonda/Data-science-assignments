# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 19:24:51 2023

@author: user
"""

import pandas as pd
df=pd.read_csv("D:\\Data science\\50_Startups.csv")
df
df.shape
df=df.rename({'R&D Spend':'RnD','Administration':'admin','Marketing Spend':'marketing','State':'state','Profit':'profit'},axis=1)
df
df.describe()
df.info()
df[df.values==0.0]
df.isnull().sum()
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(df.isnull())
df[df.duplicated()].shape
df.duplicated()
df[df.duplicated()]
#outlier detection for RnD
sns.boxplot(df['RnD'])
plt.show()
#for administration
sns.boxplot(df['admin'])
plt.show()
#for marketing spend
sns.boxplot(df['marketing'])
plt.show()
#for profit
sns.boxplot(df['profit'])
plt.show()
import numpy as np
Q1=np.quantile(df.profit,0.25)
Q3=np.quantile(df.profit,0.75)
med=np.median(df.profit)
IQR=Q3-Q1
upper_bound=Q3+(1.5*IQR)
lower_bound=Q1-(1.5*IQR)
print("Q1:",Q1,'\n',"median:",med,'\n',"Q3:", Q3,'\n',"IQR:",IQR,'\n', "upper bound:",upper_bound,'\n', "lower bound:",lower_bound )
outliers=df.profit[(df.profit<=lower_bound)|(df.profit>=upper_bound)]
print("outliers:",outliers)

#display(df[df.index.isin([49])],df.head())
plt.subplots(figsize=(9,6))
plt.subplot(131)
plt.boxplot(np.log(df['profit']))
plt.subplot(132)
plt.boxplot(np.sqrt(df['profit']))
plt.subplot(133)
plt.boxplot(np.cbrt(df['profit']))


import statsmodels.formula.api as sm
first_model=sm.ols("profit~RnD+admin+marketing",data=df).fit()
first_model.summary()

df1=df.copy()
sns.boxplot(df['profit'])
plt.title("profit before median imputation")
plt.show()
#median imputation
for i in df1['profit']:
    Q1=np.quantile(df1.profit,0.25)
    Q3=np.quantile(df1.profit,0.75)
    med=np.median(df1.profit)
    IQR=Q3-Q1
    upper_bound=Q3+(1.5*IQR)
    lower_bound=Q1-(1.5*IQR)
    if i>upper_bound or i<lower_bound:
        df1['profit']=df1['profit'].replace(i,np.median(df1['profit']))
        sns.boxplot(df1['profit'])
        plt.title('profit after median imputation')
        plt.show()
        
import statsmodels.formula.api as sm
median_model=sm.ols("profit~RnD+admin+marketing",data=df1).fit()
median_model.summary()
#mean imputation
df2=df.copy()
for i in df2['profit']:
    Q1=np.quantile(df2.profit,0.25)
    Q3=np.quantile(df2.profit,0.75)
    med=np.median(df2.profit)
    IQR=Q3-Q1
    upper_bound=Q3+(1.5*IQR)
    lower_bound=Q1-(1.5*IQR)
    if i>upper_bound or i<lower_bound:
        df2['profit']=df2['profit'].replace(i,np.mean(df1['profit']))
        sns.boxplot(df2['profit'])
        plt.title('profit after mean imputation')
        plt.show()
        
import statsmodels.formula.api as sm
mean_model=sm.ols("profit~RnD+admin+marketing",data=df2).fit()
mean_model.summary()
#removing the outliers

df3=df.copy()
def drop_outliers(data, field_name):
    iqr = 1.5*(np.percentile(data[field_name], 75) - np.percentile(data[field_name], 25))
    data.drop(data[data[field_name] > (iqr + np.percentile(data[field_name], 75))].index, inplace=True)
    data.drop(data[data[field_name] < (np.percentile(data[field_name], 25) - iqr)].index, inplace=True)

drop_outliers(df3, 'profit')
sns.boxplot(df3.profit)
plt.title('Profit after removing outliers')

#after removing outliers
clean_model=sm.ols("profit~RnD+admin+marketing",data=df3).fit()
clean_model.summary()

#eda

sns.barplot(x='state',y='profit',data=df3)
plt.show()

#histogram
df3['profit'].hist()
df3['RnD'].hist()
df3['admin'].hist()
df3['marketing'].hist()

np.sqrt(df3['profit']).hist()
np.sqrt(df3['RnD']).hist()
np.sqrt(df3['admin']).hist()
np.sqrt(df3['marketing']).hist()

np.cbrt(df3['profit']).hist()
np.cbrt(df3['RnD']).hist()
np.cbrt(df3['admin']).hist()
np.cbrt(df3['marketing']).hist()

# RD  vs  profit
import matplotlib.pyplot as plt
plt.scatter(df3['RnD'],df3['profit'])
plt.show()
# administration  vs   profit
plt.scatter(df3['admin'],df3['profit'])
plt.show()
# marketing spend  vs  profit
plt.scatter(df3['marketing'],df3['profit'])
plt.show()


df4=df3.drop('state',axis=1)
df4.head()
#calculating correlation
df4.corr()

sns.set_style(style='darkgrid')
sns.pairplot(df4)
#==========================================
Y=df3['profit']
X=df3.iloc[:,0:3]
#standardisation
from sklearn.preprocessing import StandardScaler
SS=StandardScaler()
SS_X=SS.fit_transform(X)
pd.DataFrame(SS_X)


import statsmodels.formula.api as sm
RnD_model=sm.ols("profit~RnD",data=df3).fit()
RnD_model.summary()
RnD_model.fittedvalues #predicted values
RnD_model.resid #Error values -->Y-Ypred


import statsmodels.formula.api as sm
RnD_with_marketing=sm.ols("profit~RnD+marketing",data=df3).fit()
RnD_with_marketing.summary()
RnD_with_marketing.fittedvalues #predicted values
RnD_with_marketing.resid #Error values -->Y-Ypred

import statsmodels.formula.api as sm
RnD_with_marketing_admin=sm.ols("profit~RnD+admin+marketing",data=df3).fit()
RnD_with_marketing_admin.summary()
RnD_with_marketing_admin.fittedvalues #predicted values
RnD_with_marketing_admin.resid #Error values -->Y-Ypred

model_names=['first_model','median_model','mean_model','clean_model','RnD_model','RnD_with_marketing','RnD_with_marketing_admin']
r_squared_values=[0.951,0.918,0.909,0.961,0.957,0.961,0.961]
data={"Model":model_names,"R-Squared":r_squared_values}

values=pd.DataFrame(data)
print(values)
#=====================================================
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
df=pd.read_csv("D:\\Data science\\ToyotaCorolla.csv",encoding='latin1')
df.head()
df.shape

data=df[["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]
data

data.describe()

data.info()
data[data.values==0.0]
data.isnull().sum()

data[data.duplicated()].shape
data[data.duplicated()]

data=data.drop_duplicates().reset_index(drop=True)
data[data.duplicated()]

data


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12,8))
sns.heatmap(data.isnull(),cmap='viridis')

discrete_feature=[feature for feature in data.columns if len(data[feature].unique())<20 and feature]
print('Discrete Variables Count: {}'.format(len(discrete_feature)))

continuous_feature=[feature for feature in data.columns if data[feature].dtype!='O' and feature not in discrete_feature]
print('Continuous Feature Count {}'.format(len(continuous_feature)))

import scipy.stats as stat
import pylab
def plot_data(data,feature):
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    data[feature].hist()
    plt.subplot(1,2,2)
    stat.probplot(data[feature],dist='norm',plot=pylab)
    
plot_data(data,'Price')
plt.title('Price')
plot_data(data,'Age_08_04')
plt.title('Age')
plot_data(data,'KM')
plt.title('KM')
plot_data(data,'Weight')
plt.title('Weight')

#log transformation
import numpy as np
dof=data.copy()
dof[continuous_feature]=np.log(dof[continuous_feature])

plot_data(dof,'Price')
plt.title('Price')
plot_data(dof,'Age_08_04')
plt.title('Age')
plot_data(dof,'KM')
plt.title('KM')
plot_data(dof,'Weight')
plt.title('Weight')
#square root transformation
dof=data.copy()
dof[continuous_feature]=np.sqrt(dof[continuous_feature])

plot_data(dof,'Price')
plt.title('Price')
plot_data(dof,'Age_08_04')
plt.title('Age')
plot_data(dof,'KM')
plt.title('KM')
plot_data(dof,'Weight')
plt.title('Weight')

#cuberoot transformation
dof=data.copy()
dof[continuous_feature]=np.cbrt(dof[continuous_feature])

plot_data(dof,'Price')
plt.title('Price')
plot_data(dof,'Age_08_04')
plt.title('Age')
plot_data(dof,'KM')
plt.title('KM')
plot_data(dof,'Weight')
plt.title('Weight')

#relationship b/w each independent and dependent variables
#price  vs age
plt.scatter(data['Price'],data['Age_08_04'])
plt.show()
#price  vs  
plt.scatter(data['Price'],data['KM'])
plt.show()
#price vs HP
plt.scatter(data['Price'],data['HP'])
plt.show()
#price  vs  Doors
plt.scatter(data['Price'],data['Doors'])
plt.show()
#price  vs  cc
plt.scatter(data['Price'],data['cc'])
plt.show()
#price vs geras
plt.scatter(data['Price'],data['Gears'])
plt.show()
#price vs QT
plt.scatter(data['Price'],data['Quarterly_Tax'])
plt.show()
#price  vs  weight
plt.scatter(data['Price'],data['Weight'])
plt.show()


sns.boxplot(data['Price'])
plt.title('Price')
plt.show()

sns.boxplot(data['Age_08_04'])
plt.title('Age_08_04')
plt.show()

sns.boxplot(data['KM'])
plt.title('km')
plt.show()

sns.boxplot(data['HP'])
plt.title('hp')
plt.show()

sns.boxplot(data['cc'])
plt.title('cc')
plt.show()

sns.boxplot(data['Doors'])
plt.title('doors')
plt.show()

sns.boxplot(data['Gears'])
plt.title('gears')
plt.show()

sns.boxplot(data['Quarterly_Tax'])
plt.title('qt')
plt.show()

sns.boxplot(data['Weight'])
plt.title('weight')
plt.show()

data.HP.unique()
#median imputation
df1=data.copy()
for i in data['Price']:
    q1 = np.quantile(df1.Price,0.25)
    q3 = np.quantile(df1.Price,0.75)
    med = np.median(df1.Price)
    iqr = q3 - q1
    upper_bound = q3+(1.5*iqr)
    lower_bound = q1-(1.5*iqr)
    if i > upper_bound or i < lower_bound:
        df1['Price'] = df1['Price'].replace(i, np.median(df1['Price']))
sns.boxplot(df1['Price'])
plt.title('Price after median imputation')
plt.show()
#median imputation
for i in data['Age_08_04']:
    q1 = np.quantile(df1.Age_08_04,0.25)
    q3 = np.quantile(df1.Age_08_04,0.75)
    med = np.median(df1.Age_08_04)
    iqr = q3 - q1
    upper_bound = q3+(1.5*iqr)
    lower_bound = q1-(1.5*iqr)
    if i > upper_bound or i < lower_bound:
        df1['Age_08_04'] = df1['Age_08_04'].replace(i, np.median(df1['Age_08_04']))
sns.boxplot(df1['Age_08_04'])
plt.title('Age after median imputation')
plt.show()

for i in data['KM']:
    q1 = np.quantile(df1.KM,0.25)
    q3 = np.quantile(df1.KM,0.75)
    med = np.median(df1.KM)
    iqr = q3 - q1
    upper_bound = q3+(1.5*iqr)
    lower_bound = q1-(1.5*iqr)
    if i > upper_bound or i < lower_bound:
        df1['KM'] = df1['KM'].replace(i, np.median(df1['KM']))
sns.boxplot(df1['KM'])
plt.title('KM after median imputation')
plt.show()

for i in data['Weight']:
    q1 = np.quantile(df1.Weight,0.25)
    q3 = np.quantile(df1.Weight,0.75)
    med = np.median(df1.Weight)
    iqr = q3 - q1
    upper_bound = q3+(1.5*iqr)
    lower_bound = q1-(1.5*iqr)
    if i > upper_bound or i < lower_bound:
        df1['Weight'] = df1['Weight'].replace(i, np.median(df1['Weight']))
sns.boxplot(df1['Weight'])
plt.title('Weight after median imputation')
plt.show()

import statsmodels.formula.api as smf
after_median_imputation_model = smf.ols("Price~Age_08_04+KM+Weight", data = df1).fit()
after_median_imputation_model.summary() 

#mean imputation
df2=data.copy()
for i in data['Price']:
    q1 = np.quantile(df2.Price,0.25)
    q3 = np.quantile(df2.Price,0.75)
    med = np.median(df2.Price)
    iqr = q3 - q1
    upper_bound = q3+(1.5*iqr)
    lower_bound = q1-(1.5*iqr)
    if i > upper_bound or i < lower_bound:
        df2['Price'] = df2['Price'].replace(i, np.mean(df2['Price']))
sns.boxplot(df2['Price'])
plt.title('Price after mean imputation')
plt.show()

for i in data['Age_08_04']:
    q1 = np.quantile(df2.Age_08_04,0.25)
    q3 = np.quantile(df2.Age_08_04,0.75)
    med = np.median(df2.Age_08_04)
    iqr = q3 - q1
    upper_bound = q3+(1.5*iqr)
    lower_bound = q1-(1.5*iqr)
    if i > upper_bound or i < lower_bound:
        df2['Age_08_04'] = df2['Age_08_04'].replace(i, np.mean(df2['Age_08_04']))
sns.boxplot(df2['Age_08_04'])
plt.title('Age after mean imputation')
plt.show()

for i in data['KM']:
    q1 = np.quantile(df2.KM,0.25)
    q3 = np.quantile(df2.KM,0.75)
    med = np.median(df2.KM)
    iqr = q3 - q1
    upper_bound = q3+(1.5*iqr)
    lower_bound = q1-(1.5*iqr)
    if i > upper_bound or i < lower_bound:
        df2['KM'] = df2['KM'].replace(i, np.mean(df2['KM']))
sns.boxplot(df2['KM'])
plt.title('KM after mean imputation')
plt.show()

for i in data['Weight']:
    q1 = np.quantile(df2.Weight,0.25)
    q3 = np.quantile(df2.Weight,0.75)
    med = np.median(df2.Weight)
    iqr = q3 - q1
    upper_bound = q3+(1.5*iqr)
    lower_bound = q1-(1.5*iqr)
    if i > upper_bound or i < lower_bound:
        df2['Weight'] = df2['Weight'].replace(i, np.mean(df2['Weight']))
sns.boxplot(df2['Weight'])
plt.title('Weight after mean imputation')
plt.show()

import statsmodels.formula.api as smf
after_mean_imputation_model = smf.ols("Price~Age_08_04+KM+Weight", data = df2).fit()
after_mean_imputation_model.summary() 

df3=data.copy()
def drop_outliers(data, field_name):
    iqr = 1.5*(np.percentile(data[field_name], 75) - np.percentile(data[field_name], 25))
    data.drop(data[data[field_name] > (iqr + np.percentile(data[field_name], 75))].index, inplace=True)
    data.drop(data[data[field_name] < (np.percentile(data[field_name], 25) - iqr)].index, inplace=True)
drop_outliers(df3, 'Price')
sns.boxplot(df3.Price)
plt.title('Price after removing outliers')

drop_outliers(df3, 'Age_08_04')
sns.boxplot(df3.Age_08_04)
plt.title('Age after removing outliers')

drop_outliers(df3, 'KM')
sns.boxplot(df3.KM)
plt.title('KM after removing outliers')

drop_outliers(df3, 'Weight')
sns.boxplot(df3.Weight)
plt.title('Weight after removing outliers')

removed_outlier_model = smf.ols("Price~Age_08_04+KM+Weight", data = df3).fit()
removed_outlier_model.summary()

raw_data_model = smf.ols("Price~Age_08_04+KM+Weight+HP+cc+Gears+Quarterly_Tax+Doors", data = data).fit()
raw_data_model.summary()

dataframe=data.copy()
df_log_scaled = pd.DataFrame()
df_log_scaled['Age_08_04'] = np.log(dataframe.Age_08_04)
df_log_scaled['Price'] = np.log(dataframe.Price)
df_log_scaled['KM'] = np.log(dataframe.KM)
df_log_scaled['Weight'] = np.log(dataframe.Weight)
df_log_scaled['CC'] = dataframe['cc']
df_log_scaled['Doors'] = dataframe['Doors']
df_log_scaled['HP'] = dataframe['HP']
df_log_scaled.head()


log_transformed_model = smf.ols("Price~Age_08_04+KM+HP+CC+Doors+Weight", data = df_log_scaled).fit()
log_transformed_model.summary()

df_cbrt_scaled = pd.DataFrame()
df_cbrt_scaled['Age_08_04'] = np.cbrt(dataframe.Age_08_04)
df_cbrt_scaled['Price'] = np.cbrt(dataframe.Price)
df_cbrt_scaled['KM'] = np.cbrt(dataframe.KM)
df_cbrt_scaled['Weight'] = np.cbrt(dataframe.Weight)
df_cbrt_scaled['cc'] = dataframe['cc']
df_cbrt_scaled['Quarterly_Tax'] = dataframe['Quarterly_Tax']
df_cbrt_scaled['Doors'] = dataframe['Doors']
df_cbrt_scaled['Gears'] = dataframe['Gears']
df_cbrt_scaled['HP'] = dataframe['HP']
df_cbrt_scaled.head()

cbrt_transformed_model = smf.ols("Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight", data = df_cbrt_scaled).fit()
cbrt_transformed_model.summary()

df_sqrt_scaled = pd.DataFrame()
df_sqrt_scaled['Age_08_04'] = np.sqrt(dataframe.Age_08_04)
df_sqrt_scaled['Price'] = np.sqrt(dataframe.Price)
df_sqrt_scaled['KM'] = np.sqrt(dataframe.KM)
df_sqrt_scaled['Weight'] = np.sqrt(dataframe.Weight)
df_sqrt_scaled['cc'] = dataframe['cc']
df_sqrt_scaled['Quarterly_Tax'] = dataframe['Quarterly_Tax']
df_sqrt_scaled['Doors'] = dataframe['Doors']
df_sqrt_scaled['Gears'] = dataframe['Gears']
df_sqrt_scaled['HP'] = dataframe['HP']
df_sqrt_scaled.head()

sqrt_transformed_model = smf.ols("Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight", data = df_sqrt_scaled).fit()
sqrt_transformed_model.summary()

from sklearn.preprocessing import StandardScaler

col_names = dataframe.columns
features = dataframe[col_names]

scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
df_standard_scaled = pd.DataFrame(features, columns = col_names)
df_standard_scaled.head()

standard_scaler_transformed_model = smf.ols("Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight", data = df_standard_scaled).fit()
standard_scaler_transformed_model.summary()

model_names=['after_median_imputation_model','after_mean_imputation_model','removed_outlier_model','raw_data_model','log_transformed_model','cbrt_transformed_model','sqrt_transformed_model','standard_scaler_transformed_model']
r_squared_values=[0.342,0.388,0.778,0.863,0.746,0.841,0.864,0.863]
data={"Model":model_names,"R-Squared":r_squared_values}

r_squares=pd.DataFrame(data)
print(r_squares)
