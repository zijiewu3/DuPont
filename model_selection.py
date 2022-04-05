#!/usr/bin/env python
"""Gaussian Process applied to DuPont dataset"""
#%%
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy
from sklearn.linear_model import ElasticNet, Ridge, Lasso, LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from xgboost.sklearn import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern, RationalQuadratic, WhiteKernel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
# %%
# Load dataset
Dataset = pd.read_csv("data/UD_867_formulation_training.csv")
TARGETS = ['Water_Absorption_%', 'Hardness', 'Thermal_Conductivity_(mW/m.K)']
# Reproducibility
SEED = 12345
# Training options
FEATURE_SELECTION = True # mutual information regression
ADD_ENTROPY = True
K = 24
# %%
models =    {
    # Similarity-based regressors
    'KNeighborsRegressor':{
            # 'n_neighbors': np.arange(1,3,2),
            'n_neighbors': np.arange(1,15,2),
            'weights': ['uniform','distance'],
            'p': [1,2],
            'metric': ['minkowski','chebyshev'],
    },
    'GaussianProcessRegressor':{
            'kernel':None,
            'n_restarts_optimizer':[5],
            'random_state':[SEED]
    },
    # Tree-based regressors
    'XGBRegressor':{
            'learning_rate': np.arange(0.025,0.150,0.025),
            # 'gamma':np.arange(0.05,0.45,0.05),
            # 'max_depth':np.arange(2,14,2),
            # 'min_child_weight':np.arange(1,8,1),
            'n_estimators':np.arange(10,80,5),
            # 'subsample':np.arange(0.60,0.95,0.05),
            # 'colsample_bytree':np.arange(0.60,0.95,0.05),
            'reg_alpha':np.logspace(-3,2,6), #alpha
            'reg_lambda':np.logspace(-3,2,6),#lambda
    },
    'RandomForestRegressor':{
            'n_estimators':np.arange(10,80,5),
            'criterion': ['squared_error','absolute_error','poisson'],
            'max_features': ['auto','sqrt','log2'],
            'random_state':[SEED],
    },
    # Linear regressors
    'Ridge':{
            'alpha':np.logspace(-3,2,6),
            'max_iter':[50000],
            'random_state':[SEED],
    },
    'Lasso':{
            'alpha':np.logspace(-3,2,6),
            'max_iter':[50000],
            'random_state':[SEED],
    },
    'ElasticNet':{
            'alpha':np.logspace(-3,2,6),
            'l1_ratio':np.linspace(0.1,0.9,9),
            'max_iter':[50000],
            'random_state':[SEED]
    },

            }
# %%
def set_model(name):
    """Initialization module for models to be evaluated"""
    # Similarity-based regressors
    if name=='KNeighborsRegressor':
        model = KNeighborsRegressor()
    elif name=='GaussianProcessRegressor':
        model = GaussianProcessRegressor()    
    # Tree-based regressor
    elif name=='XGBRegressor':
        model = XGBRegressor()
    elif name=='RandomForestRegressor':
        model = RandomForestRegressor()
    # Linear models
    elif name=='LinearRegression':
        model = LinearRegression()
    elif name=='Ridge':
        model = Ridge()
    elif name=='Lasso':
        model = Lasso()
    elif name=='ElasticNet':
        model = ElasticNet()
    return model
# %%
def restart_kernels(init_length_scale=1.0):
    """Function that calls kernels every time they need to be instanciated."""
    kernels = [1.0*RBF(length_scale=init_length_scale)+1.0*WhiteKernel(),
                1.0*DotProduct()+1.0*WhiteKernel(), 
                1.0*Matern(length_scale=init_length_scale, nu=0.5)+1.0*WhiteKernel(),
                1.0*Matern(length_scale=init_length_scale, nu=1.5)+1.0*WhiteKernel(),
                1.0*Matern(length_scale=init_length_scale, nu=2.5)+1.0*WhiteKernel(),  
                1.0*RationalQuadratic()+1.0*WhiteKernel()]
    return kernels
# %%
def nested_cv(X,y,models,SEED):
    """Nested cross-validation procedure for model selection"""
    metrics_name = ['r2', 'MAE', 'RMSE']
    metrics = [r2_score, mean_absolute_error, mean_squared_error]
    file_output = open('results/{}_FS-{}_AE-{}.txt'.format(y.name.split('_')[0], FEATURE_SELECTION, ADD_ENTROPY),'w')
    stats = {}
    for m_i in models:
        # stats = []
        # model = set_model(m_i)
        cv_outer = KFold(n_splits=10, shuffle=True, random_state=SEED)
        # enumerate splits
        outer_results = []
        for train_ix, test_ix in cv_outer.split(X):
            # split data
            X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
            y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]
            # configure the cross-validation procedure
            cv_inner = KFold(n_splits=5, shuffle=True, random_state=SEED)
            # define the model
            model = set_model(m_i)
            # model = RandomForestClassifier(random_state=1)
            if m_i=='GaussianProcessRegressor':
                models[m_i]['kernel'] = restart_kernels(np.ones(X.shape[1]))
            # define search space
            space = models[m_i]
            # define search
            search = GridSearchCV(model, space, scoring='r2', cv=cv_inner, refit=True, n_jobs=-1)
            # execute search
            result = search.fit(X_train, y_train)
            # get the best performing model fit on the whole training set
            best_model = result.best_estimator_
            # evaluate model on the hold out dataset
            yhat = best_model.predict(X_test)
            # evaluate the model
            acc = [metric(y_test, yhat) for metric in metrics]
            # store the result
            outer_results.append(acc)
            # report progress
            file_output.write('predictor:%s, metrics:%s, best params:%s\n' % (m_i, acc, result.best_params_))
            # print('predictor:%s, metrics:%s, best params:%s',acc,' best params:',result.best_params_)
	        # print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
        outer_df = pd.DataFrame(outer_results)
        stats[m_i] = dict(zip(metrics_name,list(zip(outer_df.mean().values,outer_df.std().values))))
    file_output.close()
    return stats
# %%
def std_features(X):
    """Standardization function"""
    # Standardization
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    return (X - mu)/sigma
# %%
# feature selection
def select_features(X, y, k):
    """Feature selection function"""
    # configure to select all features
    fs = SelectKBest(score_func=mutual_info_regression, k=k)
    # learn relationship from training data
    fs.fit(X, y)
    # take features out
    X_fs = pd.DataFrame(fs.transform(X), columns=fs.feature_names_in_[fs.get_support()])
    return X_fs
# %%
# Select only features from dataset
X_train = Dataset.drop(columns=['name']+TARGETS)
# add entropy as a feature
if ADD_ENTROPY:
    X_train['entropy']=scipy.stats.entropy(X_train, axis=1)
# Select only targets from dataset
Y_train = Dataset[TARGETS]
# Standardization
X_train = std_features(X_train)
# Main loop to evaluate multiple models
results = {}
for target in TARGETS:
    y_train = Y_train[target]
    if FEATURE_SELECTION:
        X_train = select_features(X_train, y_train, K)
    results[target]=nested_cv(X_train,y_train,models,SEED)

RESULTS_PATH = 'results/summary_results_FS-{}_AE-{}.csv'.format(FEATURE_SELECTION, ADD_ENTROPY)
pd.DataFrame.from_dict({(i,j): results[i][j] 
                           for i in results.keys() 
                           for j in results[i].keys()},
                       orient='index').to_csv(RESULTS_PATH)
# %%

# %%
