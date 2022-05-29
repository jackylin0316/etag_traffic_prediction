# ====================================================create a 7 months training X
import pandas as pd

df=pd.read_csv('2020_AFE_0516.csv')
df=df.drop(df.columns[0], axis=1)
df=df[df['V-Type']==31]
df=df.drop('Year', axis=1)
df=df[df['Holiday']!=2]
df=df.reset_index(drop=True)
df.head()
# 刪掉宜蘭
# 北上刪掉 05F0438N-05FR143N 留 05F0438N-05F0309N
df=df[df['Segment']!='05F0438N-05FR143N']


dfc = pd.read_csv('etag_climb.csv')
dfc['Climb'].shape
df = df.merge(dfc,left_on = 'Segment',right_on = 'Segment1',how='left')
df['Climb'] = df['Climb'].fillna(0).astype(int)
df['High'] = df['Segment'].apply(lambda x : 1 if ((x[2]=='H') or (x[11]=='H')) else 0)

df['seg']=df['Segment']
temp=df['seg'].to_frame().drop_duplicates(keep='first')
temp.shape
temp=temp.reset_index(drop=True)
temp['cat']=temp['seg'].astype('category').cat.codes
temp.head()
df=df.reset_index(drop=True)
df=pd.merge(df,temp,on='seg')
temp


# final=pd.DataFrame(columns=['seg','Direction','Hour', 'Month', 'Wday', 'Holiday', 'cat'])
final=pd.DataFrame(columns=['seg','cat', 'Direction','Hour', 'Month', 'Wday', 'Holiday'])

for month in range(6,13):
    row=[]
    data=pd.DataFrame(columns=['seg','cat','Direction','Hour', 'Month', 'Wday', 'Holiday'])
    for i in range(len(temp)):
        row.append(temp.loc[i,'seg'])
        row.append(temp.loc[i,'cat'])
        if temp.loc[i,'seg'][7]=='N':
            row.append(1)
        else:
            row.append(0)
        for wday in range(7):
            for h in range(24):
                row.append(h)
                row.append(month)
                row.append(wday)
                if wday==0 or wday==6:
                    row.append(1)
                else:
                    row.append(0)
                data.loc[len(data)]=row
                row=row[:3]
        row=[] 
    final=pd.concat([final, data], axis=0)
final

    
final.head()
final = final.merge(dfc,left_on = 'seg',right_on = 'Segment1',how='left')
final['Climb'] = final['Climb'].fillna(0).astype(int)
final['High'] = final['seg'].apply(lambda x : 1 if ((x[2]=='H') or (x[11]=='H')) else 0)
final.to_csv('months_climb.csv')

final
final.to_csv('months.csv')

# seven: 365736
# 499800   

# ==============================================model
import pandas as pd
def speed_(x):
    if x<50.0:
        return 0
    elif (x>=50 and x<=80):
        return 1
    else:
        return 2
       
df['speed_class']=df['Speed'].apply(lambda x: speed_(x))
df['speed_class'].value_counts()



# X=df[['Direction','Hour', 'Month', 'Wday', 'Holiday', 'cat']]
X=df[['Direction','Hour', 'Month', 'Wday', 'Holiday', 'cat', 'Climb', 'High']]


y=df[['speed_class']]
X.head()
y.head()
print(X.shape)
print(y.shape)


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn import metrics
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from scipy.stats import uniform, randint
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV
# X=X.to_pandas()
# y=y.to_pandas()


X=X.to_numpy()
y=y.to_numpy()
# ====================================
clf_xgb = xgb.XGBClassifier(objective = 'binary:logistic')
param_dist = {'n_estimators': randint(150, 1000),
              'learning_rate': uniform(0.01, 0.59),
              'subsample': uniform(0.3, 0.6),
              'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
              'colsample_bytree': uniform(0.5, 0.4),
              'min_child_weight': [1, 2, 3, 4]
             }


kfold_5 = KFold(shuffle = True)

clf = RandomizedSearchCV(clf_xgb, 
                         param_distributions = param_dist,
                         cv = kfold_5,  
                         n_iter = 3, 
                         scoring = 'roc_auc', 
                         error_score = 0, 
                         verbose = 3, 
                         n_jobs = -1)


X_train_xg, X_test_xg, y_train_xg, y_test_xg = train_test_split(X, y, test_size=0.1, random_state=20)

clf.fit(X_train_xg, y_train_xg)

def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

report_best_scores(clf.cv_results_, 1)
# ====================================
# X_train_xg, X_test_xg, y_train_xg, y_test_xg = train_test_split(X, y, test_size=0.1, random_state=20)
X_train_xg, X_test_xg, y_train_xg, y_test_xg = train_test_split(X, y, test_size=0.1, random_state=20)

# ==================================classfier
model = xgb.XGBClassifier(colsample_bytree=0.6593330974815904, learning_rate= 0.2796339926807502, max_depth= 5, min_child_weight= 1, n_estimators= 872, subsample=0.6110079699316302)


# import joblib
# model = joblib.load('model_0515(0 (1).9487)')

model.fit(X_train_xg, y_train_xg)
y_pred_xg = model.predict(X_test_xg)
mse = mean_squared_error(y_test_xg, y_pred_xg)
r2 = r2_score(y_test_xg, y_pred_xg)
xg_acc = metrics.accuracy_score(y_test_xg, y_pred_xg)
print('accuracy: {}'.format(xg_acc))

df
# joblib.dump(model, 'whole_year')



# ======================================
# train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(model, X, y, cv=5,return_times=True)

# plt.plot(train_sizes,np.mean(train_scores,axis=1))
# plt.ylabel('accuracy')
# plt.xlabel('data in millions')
# ======================================


X=pd.read_csv('months.csv')
# X=pd.read_csv('months_X_1.csv')
X=X.drop(X.columns[0], axis=1)
X
final=X
X=X[['Direction','Hour', 'Month', 'Wday', 'Holiday', 'cat', 'Climb', 'High']]
X
pre=model.predict(X)
predict=pd.DataFrame(pre, columns=['predict'])
predict
final=pd.concat([final,predict], axis=1)
final

# =============================check
len(final['seg'].unique())
final['predict'].value_counts()
# =============================

final.to_csv('05_24_whole_year(0.96).csv')

test=final




# ==================================adaboost
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(DecisionTreeClassifier())
model.fit(X_train_xg, y_train_xg)
# model.fit(X_train_xg, y_train_xg.values.ravel()) (10,1)==>(10,)
y_pred_xg = model.predict(X_test_xg)
mse = mean_squared_error(y_test_xg, y_pred_xg)
r2 = r2_score(y_test_xg, y_pred_xg)
xg_acc = metrics.accuracy_score(y_test_xg, y_pred_xg)
print('accuracy: {}'.format(xg_acc))
# print('r2: {}'.format(r2))
# print('mse: {}'.format(mse))


# ===================================RandomForest
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train_xg, y_train_xg)
# model.fit(X_train_xg, y_train_xg.values.ravel())
y_pred_xg = model.predict(X_test_xg)
mse = mean_squared_error(y_test_xg, y_pred_xg)
r2 = r2_score(y_test_xg, y_pred_xg)
xg_acc = metrics.accuracy_score(y_test_xg, y_pred_xg)
print('accuracy: {}'.format(xg_acc))
print('r2: {}'.format(r2))
print('mse: {}'.format(mse))

# ==================================filter
segment=pd.read_csv('seg.csv')
ls=segment['seg'].tolist()
final=final[final['seg'].isin(ls)]
final=final.reset_index(drop=True)
final
# ========================================補 19 segments to 330 segments
dif=['01F0492S-01F0511S',
 '01F0467S-01F0492S',
 '01F1699S-01F1774S',
 '01F2425S-01F2472S',
 '01F2472S-01F2514S',
 '01F1699N-01F1664N',
 '03F0783N-03F0746N',
 '05F0001N-03F4259N',
 '01F2472N-01F2425N',
 '05F0309S-05FR113S',
 '01F1774N-01F1699N',
 '05FR143N-05F0528N',
 '01F0492N-01F0467N',
 '01F2514N-01F2472N',
 '01F1664S-01F1699S',
 '01F0511N-01F0492N',
 '03F0698S-03F0746S',
 '03F0746S-03F0783S',
 '03F0746N-03F0698N']
 
for i in range(19):
    s1=final[final['seg'].str[9:]==dif[i][:8]]
    s1.loc[:, 'seg']=dif[i]
    s2=final[final['seg'].str[:8]==dif[i][9:]]
    s2.loc[:, 'seg']=dif[i]
    if s1.shape[0]==0:
        final=pd.concat([final, s2], axis=0)
    else:
        final=pd.concat([final, s1], axis=0)

final=final.reset_index(drop=True)
final

len(final['seg'].unique())

