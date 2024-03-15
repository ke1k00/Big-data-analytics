#!/usr/bin/env python
# coding: utf-8

# ## Data Exploration and Visualization

# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

housing = pd.read_csv('housing.csv')
print("Housing head:")
print(housing.head())
print()

print("Housing info:")
housing.info()
print()

print("Ocean proximity sub categories:")
print(housing["ocean_proximity"].value_counts())
print()

print("Housing stats:")
print(housing.describe())
print()

print("Visualisation:")
housing.hist(bins=50, figsize=(30, 30))
housing.plot(kind="scatter", 
             x="longitude", 
             y="latitude", 
             alpha=0.2, 
             s=housing["population"]/100, 
             label="population", 
             c="median_house_value", 
             cmap=plt.get_cmap("jet"))


# ## Correlation Analysis

# In[2]:


corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# ## Data Cleaning and Preprocessing

# In[5]:


housing_na = housing.dropna(subset=["total_bedrooms"])
dummies = pd.get_dummies(housing_na.ocean_proximity)
housing_na_dummies = pd.concat([housing_na, dummies], axis='columns')
housing_clean = housing_na_dummies.drop(["ocean_proximity", "ISLAND"], axis="columns")


# ## Model Training and Evaluation

# In[7]:


X = housing_clean.drop(columns=["median_house_value"])
y = housing_clean["median_house_value"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1984)

from sklearn.linear_model import LinearRegression
OLS = LinearRegression()
OLS.fit(X_train, y_train)

print("Intercept is " + str(OLS.intercept_))
print()
print("The set of coefficients are:")
print(OLS.coef_)
print()
print("The R-squared value is " + str(OLS.score(X_train, y_train)))

y_pred = OLS.predict(X_test)


# ## Visualization of Model Performance

# In[8]:


performance = pd.DataFrame({'PREDICTIONS': y_pred, 'ACTUAL VALUES': y_test})
performance["error"] = performance['ACTUAL VALUES'] - performance['PREDICTIONS']
performance.reset_index(drop=True, inplace=True)
performance.reset_index(inplace=True)

fig = plt.figure(figsize=(10, 5))
plt.bar('index', 'error', data=performance[:50], color='black', width=0.3)
plt.xlabel("Observations")
plt.ylabel("Residuals")
plt.show()


# ## Advanced Model Summary

# In[9]:


import statsmodels.api as sm
X_train = sm.add_constant(X_train)
nicerOLS = sm.OLS(y_train, X_train).fit()
nicerOLS.summary()

