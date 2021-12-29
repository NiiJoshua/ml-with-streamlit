# import libraries
import pandas as pd
import numpy as np
from pandas.core.tools.datetimes import Scalar
from sklearn.linear_model import Ridge, LinearRegression, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.model_selection import cross_val_score, RepeatedKFold, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle
import warnings

from sklearn.utils import shuffle

warnings.filterwarnings('ignore')

# loading the data
data = pd.read_excel('Data.xlsx')
# split data into train and test
X_train = data.iloc[0:10, 0:5]
Y_train = data.iloc[0:10, 5:18]

X_test = data.iloc[10:,0:5]
Y_test = data.iloc[10:,5:18] 

# Evaluate algorithms
models = []
models.append(('LR',LinearRegression()))
models.append(('R',Ridge()))
models.append(('LASSO',Lasso()))
models.append(('EN',ElasticNet()))
models.append(('KNN',KNeighborsRegressor()))
models.append(('CART',DecisionTreeRegressor()))

#Evaluate models in turn
results = []
names = []
scoring = 'neg_mean_absolute_error'

for name, model in models:
    kfold = KFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Evaluate algorithms on standardized dataset
pipelines = []
pipelines.append(('ScaledLR',Pipeline([('Scaler',StandardScaler()),('LR',LinearRegression())])))
pipelines.append(('ScaledR',Pipeline([('Scaler',StandardScaler()),('R',Ridge())])))
pipelines.append(('ScaledLASSO',Pipeline([('Scaler',StandardScaler()),('LASSO',Lasso())])))
pipelines.append(('ScaledEN',Pipeline([('Scaler',StandardScaler()),('EN',ElasticNet())])))
pipelines.append(('ScaledKNN',Pipeline([('Scaler',StandardScaler()),('KNN',KNeighborsRegressor())])))
pipelines.append(('ScaledCART',Pipeline([('Scaler',StandardScaler()),('CART',DecisionTreeRegressor())])))

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

# Lasso performs better
# Tune Lasso Regressor
# defing model
model = Lasso()
# define model evaluation method
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)

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

# Save final model
with open('tuned_pkl','wb') as files:
    pickle.dump(final_model, files)

# make predictions
x_test_scaled = scaler.transform(X_test)
y_pred = final_model.predict(x_test_scaled)

# Evaluate performance
print("MAE", mean_absolute_error(Y_test, y_pred))
print("MSE", mean_squared_error(Y_test, y_pred))
print("RMSE", np.sqrt(mean_squared_error(Y_test, y_pred)))

pred_columns = ['Back Ramp', 'Centre Base', 'Front Ramp', 'Back Wall', 'Left Wall','Right Wall', 'Roof Beams',
'Lintel Beam', 'Door Shaft','Door Fabrication', 'Heat Shield', 'Door Surround Casting','Refractory']
# save output to dataframe
output_df = pd.DataFrame(y_pred, columns=pred_columns)
output_df
