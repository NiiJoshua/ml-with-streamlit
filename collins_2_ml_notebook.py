#!/usr/bin/env python
# coding: utf-8

# # Machine Learning model for MIL
# 
# The work below examines the data from Mechatherm International Limited and experimentation of linear and nonlinear models to determine which one generalizes well on the data.

# ## Module imports

# In[1]:


# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.tools.datetimes import Scalar
from sklearn.linear_model import Ridge, LinearRegression, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.model_selection import cross_val_score, RepeatedKFold, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import pickle
import warnings

from sklearn.utils import shuffle

warnings.filterwarnings('ignore')


# ## Loading and splitting data

# In[2]:


# loading the data
data = pd.read_excel('Data.xlsx')
# split data into train and test
X_train = data.iloc[0:10, 0:5]
Y_train = data.iloc[0:10, 5:18]

X_test = data.iloc[10:,0:5]
Y_test = data.iloc[10:,5:18] 


# ## Inital model evaluation 

# In[3]:


# create a list for linear models
l_models = []
l_models.append(('LR',LinearRegression()))
l_models.append(('R',Ridge()))
l_models.append(('LASSO',Lasso()))
l_models.append(('EN',ElasticNet()))

# create for nonlinear models
nl_models = []
nl_models.append(('DT',DecisionTreeRegressor()))
nl_models.append(('RF',RandomForestRegressor()))
nl_models.append(('eT',ExtraTreesRegressor()))


# In[4]:


#Evaluate linear models in turn
results = []
names = []
scoring = 'neg_mean_absolute_error'

for name, model in l_models:
    kfold = KFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[5]:


#Evaluate linear models in turn
results = []
names = []
scoring = 'neg_mean_absolute_error'

for name, model in nl_models:
    kfold = KFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# ### Observations from the initial model evaluation (before standardization)
# 
# * The linear models performed better in general compared to the performance of the nonlinear models. 
# * Linear regression and Ridge regression outperformed all the linear models.
# * Elastic Net did better than all the nonlinear modelsbut it's cross validated mean is worse than the worse performing linear model.
# 
# 
# 

# In[6]:


# Evaluate algorithms on standardized dataset
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
pipelines = []
# Linear regression models
pipelines.append(('ScaledLR',Pipeline([('Scaler',StandardScaler()),('LR',LinearRegression())])))
pipelines.append(('ScaledR',Pipeline([('Scaler',StandardScaler()),('R',Ridge())])))
pipelines.append(('ScaledLASSO',Pipeline([('Scaler',StandardScaler()),('LASSO',Lasso())])))
pipelines.append(('ScaledEN',Pipeline([('Scaler',StandardScaler()),('EN',ElasticNet())])))
# nonlinear regression models
pipelines.append(('ScaledDT',Pipeline([('Scaler',StandardScaler()),('DT',DecisionTreeRegressor())])))
pipelines.append(('ScaledRF',Pipeline([('Scaler',StandardScaler()),('RF',RandomForestRegressor())])))
pipelines.append(('ScaledET',Pipeline([('Scaler',StandardScaler()),('ET',ExtraTreesRegressor())])))


# In[7]:


# Evaluate each model in turn
results = []
names = []
scoring = 'neg_mean_absolute_error'

for name, model in pipelines:
    kfold = KFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# ### Observations after standardising the data
# 
# * Linear regression maintained it's values compared to its performance before standardisation. It could be said that it generalized well
# * Ridge regression which was close to the Linear regression did not do so well after standardisation
# * Lasso regression outperformed all the models and hence the optimal model for the data
# * Elastic Net performed poorly of all the linear models
# * The nonlinear models did not improve significantly. This can be attributed to the volume of training data
# 
# 
# 
# 
# ## Hyperparameter tuning for Lasso Regression

# In[8]:


# Lasso performs better
# Tune Lasso Regressor
# defing model
model = Lasso()
# define model evaluation method
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)


# In[9]:


# define grid
grid = dict()
grid['alpha'] = np.arange(0,1,0.01)

# define search
search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

# perform the search
sc = StandardScaler()
X_scaled = sc.fit_transform(X_train)
results = search.fit(X_scaled, Y_train)

# summerize
print('MAE: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)


# In[10]:


# Build Final Model
# initialize scaler
scaler = sc.fit(X_train)
# saving scaler for deployment
with open('scaler_pkl','wb') as files:
    pickle.dump(scaler, files)
#  transform train data with scaler
X_scaled = sc.transform(X_train)
final_model = Lasso(alpha=0.99)
final_model.fit(X_scaled, Y_train)


# In[11]:


# Save final model
with open('tuned_pkl','wb') as files:
    pickle.dump(final_model, files)


# In[12]:


# make predictions
x_test_scaled = scaler.transform(X_test)
y_pred = final_model.predict(x_test_scaled)


# In[13]:


# Evaluate performance
print("MAE", mean_absolute_error(Y_test, y_pred))
print("MSE", mean_squared_error(Y_test, y_pred))
print("RMSE", np.sqrt(mean_squared_error(Y_test, y_pred)))
print("R_score", r2_score(Y_test, y_pred))

pred_columns = ['Back Ramp', 'Centre Base', 'Front Ramp', 'Back Wall', 'Left Wall','Right Wall', 'Roof Beams',
'Lintel Beam', 'Door Shaft','Door Fabrication', 'Heat Shield', 'Door Surround Casting','Refractory']
# save output to dataframe
output_df = pd.DataFrame(y_pred, columns=pred_columns)
output_df


# ## Performance of the Lasso regression
# 
# * Using a grid search, the best penalty paramter was 0.99
# * After implementation, the model was evaluated on the Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Square Error (RMSE) and the r-squared
# * values for the final evaluation are on the hihger side which could be becuase of the amount of training data.

# In[14]:


data


# In[15]:


data.describe()


# In[16]:


data.info()


# In[17]:


X_train


# In[18]:


Y_train


# In[19]:


coeffs = final_model.coef_ # extracting the coefficients from the lasso regression
type(coeffs)
ls = np.array(coeffs).tolist() # convert the coefficient results from numpy to a list


# In[20]:


# intercepts for the each output column
intercept = final_model.intercept_
intercept


# In[21]:


data.corr(method='pearson')


# ## Correlation coefficients
# 
# * The correlation coefficients were based on the pearson correlation.
# * From the correlation values, it can be concluded that the values are highly correlated
