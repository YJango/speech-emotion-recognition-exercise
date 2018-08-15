import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV
epsilon = 1e-15

from sklearn import svm
from lightgbm.sklearn import LGBMRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import linear_model
from sklearn import gaussian_process
from sklearn.decomposition import PCA
# get 1400 data ,X: features, Y: label
def get_X_Y(feature_set_name='IS09'):
    PATH = '/home/jyu/haoweilai/'
    df_train = pd.read_csv(PATH+'dataframe/train_%s.csv' %feature_set_name) 
    mean,std = np.load(PATH+'dataframe/%s_mean_std.npy' %feature_set_name)
    train = np.array((df_train.set_index('id')-mean)/(std+epsilon))
    X = train[:,:-1]
    Y = train[:,-1]
    return X,Y

# GridSearchCV
def tuning(emsr, param_grid, k_fold = 7, scores = ['neg_mean_squared_error','r2']):
    for score in scores:
        gs = GridSearchCV(emsr, param_grid, cv=k_fold, scoring=score)
        gs.fit(X, Y)
        print('%s: %s' %(score,np.abs(gs.best_score_)))
# CV
def cv_result(emsr, k_fold = 7, scores = ['neg_mean_squared_error','r2']):
    for score in scores:
        cv = cross_val_score(emsr, X, Y, cv=k_fold, scoring=score)
        print('%s: %s' %(score,np.abs(cv).mean()))
'''      
for feature_set_name in ['IS09','IS10','IS13','IS16']:
    X,Y= get_X_Y(feature_set_name)
    
    print('%s %s ' %(feature_set_name,'SVR'))
    emsr = svm.SVR()
    cv_result(emsr)
    #param_grid = [{'C': [1, 10,100]}]
    #tuning(emsr, param_grid)
    
    print('\n%s %s ' %(feature_set_name,'Gbdt'))
    emsr = LGBMRegressor(num_leaves =15, learning_rate=0.02, n_estimators=400)
    cv_result(emsr)
    
    #print('\n%s %s' %(feature_set_name,'AdaBoost'))
    #emsr = AdaBoostRegressor(n_estimators=100)
    #cv_result(emsr)  
    
    print('\n%s %s' %(feature_set_name,'Ridge Regression'))
    emsr = linear_model.Ridge (alpha = .5)
    cv_result(emsr)
    
    print('\n%s %s' %(feature_set_name,'Bayesian Ridge Regression'))
    emsr = linear_model.BayesianRidge()
    cv_result(emsr)'''


Xs=[]
for feature_set_name in ['IS10','IS13']:
    X,Y = get_X_Y(feature_set_name)
    Xs.append(X)
X = np.hstack(Xs)
print(X.shape,Y.shape)

print('\n%s %s ' %(feature_set_name,'Gbdt'))
emsr = LGBMRegressor(num_leaves =15, learning_rate=0.02, n_estimators=400)
cv_result(emsr)