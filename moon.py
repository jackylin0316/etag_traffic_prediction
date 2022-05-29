# ====================================================create a 7 months training X
import pandas as pd

df=pd.read_csv('holidays_moon_0524_AFE.csv')
df=df.drop(df.columns[0], axis=1)
df
df=df[df['V-Type']==31]
df=df.drop('year', axis=1)
df=df[df['Holiday']!=2]
df=df.reset_index(drop=True)
df.head()

dfc = pd.read_csv('etag_climb.csv')
dfc['Climb'].shape
df = df.merge(dfc,left_on = 'Segment',right_on = 'Segment1',how='left')
df['Climb'] = df['Climb'].fillna(0).astype(int)
df['High'] = df['Segment'].apply(lambda x : 1 if ((x[2]=='H') or (x[11]=='H')) else 0)

df['seg']=df['Segment']
temp=pd.read_csv('temp.csv')
temp=temp.drop(temp.columns[0], axis=1)
ls=temp['seg'].tolist()
df=df[df['seg'].isin(ls)]
df=df.reset_index(drop=True)
# temp=df['seg'].to_frame().drop_duplicates(keep='first')
# temp.shape
# temp=temp.reset_index(drop=True)
# temp['cat']=temp['seg'].astype('category').cat.codes
# temp.head()
df=pd.merge(df,temp,on='seg')
temp
df.shape


# 1/20(小年夜)-1/29(日)
# (五到日)

# final=pd.DataFrame(columns=['seg','Direction','Hour', 'Month', 'Wday', 'Holiday', 'cat'])

final=pd.DataFrame(columns=['seg','cat', 'Direction','Hour', 'Month', 'Wday', 'Order'])
temp
row=[]



    for i in range(len(temp)):
        row.append(temp.loc[i,'seg'])
        row.append(temp.loc[i,'cat'])
        if temp.loc[i,'seg'][7]=='N':
            row.append(1)
        else:
            row.append(0)
        order=-1
        for wday in [4,5,6,0]:
            for h in range(24):
                row.append(h)
                row.append(1)
                row.append(wday)
                row.append(order)
                final.loc[len(final)]=row
                row=row[:3]
            order+=1
        row=[] 


final
final = final.merge(dfc,left_on = 'seg',right_on = 'Segment1',how='left')
final['Climb'] = final['Climb'].fillna(0).astype(int)
final['High'] = final['seg'].apply(lambda x : 1 if ((x[2]=='H') or (x[11]=='H')) else 0)
final.to_csv('moon_climb.csv')
# final.to_csv('months_X.csv')  
df

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
df['Order'].unique()

# X=df[['Direction','Hour', 'Month', 'Wday','cat', 'Order']]
X=df[['Direction','Hour', 'Month', 'Wday', 'cat', 'Order','Climb', 'High']]


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

# X=X.to_pandas()
# y=y.to_pandas()


X=X.to_numpy()
y=y.to_numpy()

X_train_xg, X_test_xg, y_train_xg, y_test_xg = train_test_split(X, y, test_size=0.1, random_state=20)




# ==================================classfier
model = xgb.XGBClassifier()

# import joblib
# model = joblib.load('model_0515(0 (1).9487)')


model.fit(X_train_xg, y_train_xg)
y_pred_xg = model.predict(X_test_xg)
mse = mean_squared_error(y_test_xg, y_pred_xg)
r2 = r2_score(y_test_xg, y_pred_xg)
xg_acc = metrics.accuracy_score(y_test_xg, y_pred_xg)
print('accuracy: {}'.format(xg_acc))
# print('r2: {}'.format(r2))
# print('mse: {}'.format(mse))




# joblib.dump(model, 'whole_year')



# ======================================
# train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(model, X, y, cv=5,return_times=True)

# plt.plot(train_sizes,np.mean(train_scores,axis=1))
# plt.ylabel('accuracy')
# plt.xlabel('data in millions')
# ======================================

# X=pd.read_csv('months_X.csv')
X=pd.read_csv('moon_climb.csv')
X=X.drop(X.columns[0], axis=1)
X
final=X
# X=X[['Direction','Hour', 'Month', 'Wday', 'cat', 'Order']]
X=X[['Direction','Hour', 'Month', 'Wday', 'cat', 'Order','Climb', 'High']]
X
pre=model.predict(X)
predict=pd.DataFrame(pre, columns=['predict'])
predict
final=pd.concat([final,predict], axis=1)

# =============================check
len(final['seg'].unique())
final['predict'].value_counts()


final.to_csv('0524_moon.csv')

final
