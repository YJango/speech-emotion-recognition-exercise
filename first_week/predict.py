import numpy as np
import pandas as pd
from lightgbm.sklearn import LGBMRegressor
epsilon = 1e-15
PATH = '/home/jyu/haoweilai/'
np.set_printoptions(suppress=True)
def get_train(feature_set_name='IS09'):
    df_train = pd.read_csv(PATH+'dataframe/train_%s.csv' %feature_set_name) 
    mean,std = np.load(PATH+'dataframe/%s_mean_std.npy' %feature_set_name)
    train = (df_train.set_index('id')-mean)/(std+epsilon)
    x_train, y_train = train, train.pop('label')
    return x_train, y_train
def get_test(feature_set_name='IS09'):
    epsilon = 1e-15
    PATH = '/home/jyu/haoweilai/'
    df_test = pd.read_csv(PATH+'dataframe/test_%s.csv' %feature_set_name) 
    mean,std = np.load(PATH+'dataframe/%s_mean_std.npy' %feature_set_name)
    test = (df_test.set_index('id')-mean[:-1])/(std[:-1]+epsilon)
    return test
x_trains=[]
x_tests=[]
mean,std = np.load('/home/jyu/haoweilai/dataframe/IS09_mean_std.npy')
for fsn in ['IS09','IS10','IS13','IS16']:
    x_train, y_train = get_train(feature_set_name=fsn)
    x_test = get_test(feature_set_name=fsn)
    x_trains.append(x_train)
    x_tests.append(x_test)
x_train = np.hstack(x_trains)
x_test = np.hstack(x_tests)
print('train',x_train.shape)
print('test',x_test.shape)

clf = LGBMRegressor(num_leaves =15, learning_rate=0.02, n_estimators=400).fit(x_train, y_train)
p_test = clf.predict(x_test)*(std[-1]+epsilon)+mean[-1]
X_test =  pd.read_csv(PATH+'dataframe/test_IS09.csv') 
df = pd.DataFrame({'id':X_test['id'],'P':p_test},columns=['id','P']).sort_values('id').set_index('id')
with open('submission_sample','w') as f:
    for i in range(11401,12001):
        f.write('%s %s\n' %(i,df.loc[i].tolist()[0]))
print(df)