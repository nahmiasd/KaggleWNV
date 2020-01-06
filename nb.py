#%%
import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit,cross_val_score
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn.preprocessing import LabelEncoder
from math import radians, cos, sin, asin, sqrt
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
np.random.seed(42)
#%%
def add_weather_info_nearest(df):
        newrows=[]
        for index,row in df.iterrows():
                lat,lon=row['Latitude'],row['Longitude']
                distances=[haversine(lon,lat,st[0],st[1]) for st in StationsLatLong]
                if distances[0]<distances[1]:
                        newrows.append(weather_1[weather_1['Date']==row['Date']].iloc[0])
                else:
                        newrows.append(weather_2[weather_2['Date']==row['Date']].iloc[0])
        weather_nearest=pd.DataFrame(newrows)
        weather_nearest.drop('Date',axis=1,inplace=True)
        weather_nearest.index=pd.RangeIndex(start=0,stop=weather_nearest.shape[0],step=1)
        ret=pd.concat([df,weather_nearest],axis=1)
        return ret
#%%
def add_weather_info_all(df,weather_1,weather_2):
        global StationsLatLong
        weather_1=weather_1.rename(lambda x:"st1_"+x if x!="Date" else x,axis=1)
        weather_2=weather_2.rename(lambda x:"st2_"+x if x!="Date" else x,axis=1)
        wt=weather_1.merge(weather_2,on='Date')
        df=df.merge(wt,on="Date")
        lats=df['Latitude'].values
        longs=df['Longitude'].values
        dist1=[haversine(longs[i],lats[i],StationsLatLong[0][0],StationsLatLong[0][1]) for i in range(lats.shape[0])]
        dist2=[haversine(longs[i],lats[i],StationsLatLong[1][0],StationsLatLong[1][1]) for i in range(lats.shape[0])]
        df['dist1']=dist1
        df['dist2']=dist2
        return df
#%%
def cast_weather_numeric(df):
        for col in df.columns:
                if col=="Date":
                        continue
                df[col]=pd.to_numeric(df[col])
#%%
def add_date_features(df):
        temp=pd.DatetimeIndex(df.Date)
        df['Month']=temp.month
        df['WeekOfYear']=temp.weekofyear
        df['DayOfYear']=temp.dayofyear
        df.drop('Date',axis=1,inplace=True)

#%%
def add_dummies(train,test,columns=None):
        train_objs_num=len(train)
        both=pd.concat([train,test],axis=0)
        both.drop(['Id'],axis=1,inplace=True)
        both=pd.get_dummies(both,columns=columns)
        train_d=both[:train_objs_num]
        test_d=both[train_objs_num:]
        return train_d,test_d
#%%
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r
#%%
def convert_categorical(train, test, columns):
    lbl = LabelEncoder()
    for col in columns:
        lbl.fit(list(train[col].values) + list(test[col].values))
        train[col] = lbl.transform(train[col].values)
        test [col] = lbl.transform(test [col].values)
#%%
def aggregate_num_mosquitos(train,test):
        num_by_trap = pd.groupby(train[['Trap', 'NumMosquitos', 'WnvPresent']], 'Trap').agg('sum')
        num_by_trap['trap_percent_of_all_mosquitos'] = num_by_trap['NumMosquitos']/sum(num_by_trap.NumMosquitos)
        num_by_trap['trap_percent_with_wnv'] = num_by_trap.WnvPresent/num_by_trap.NumMosquitos
        num_by_trap.reset_index(inplace=True)
        map_mosq_weight = {t:v for t, v in zip(num_by_trap.Trap.values, num_by_trap['trap_percent_of_all_mosquitos'].values)}
        map_wnv_weight = {t:v for t, v in zip(num_by_trap.Trap.values, num_by_trap['trap_percent_with_wnv'].values)}
        train['trap_mosq_rate'] = train.Trap.map(map_mosq_weight)
        train['trap_wnv_rate'] = train.Trap.map(map_wnv_weight)
        test['trap_mosq_rate'] = test.Trap.map(map_mosq_weight).fillna(0)
        test['trap_wnv_rate'] = test.Trap.map(map_wnv_weight).fillna(0)
        return train,test
#%%
train = pd.read_csv('train.csv',parse_dates=['Date'])
test = pd.read_csv('test.csv',parse_dates=['Date'])
sample = pd.read_csv('sampleSubmission.csv')
weather = pd.read_csv('weather.csv',parse_dates=['Date'])
train,test = aggregate_num_mosquitos(train,test)
#%%
todrop=['AddressAccuracy','AddressNumberAndStreet','Address','WnvPresent','NumMosquitos'] #Street
y=train.WnvPresent.values
num_mosq=train.NumMosquitos.values
train.drop(todrop,axis=1,inplace=True)
todrop.remove('WnvPresent')
todrop.remove('NumMosquitos')
test.drop(todrop,axis=1,inplace=True)

#%%
weather = weather.replace('M', np.NaN)
weather = weather.replace('-',np.NaN)
weather = weather.replace('T', np.NaN)
weather = weather.replace(' T', np.NaN)
weather = weather.replace('  T', np.NaN)
#%%
StationsLatLong=[(-87.933,41.995),(-87.752,41.786)]
#%%
#weather.drop('Water1',inplace=True,axis=1)
todrop=['CodeSum','Depart','Sunrise','Sunset','Depth','SnowFall']
weather.drop(todrop,axis=1,inplace=True)
cast_weather_numeric(weather)
weather.dropna(axis=1,inplace=True,thresh=1000)
weather_1 = weather[weather['Station']==1]
weather_2 = weather[weather['Station']==2]
weather_1.fillna(method='ffill',inplace=True)
weather_2.fillna(method='ffill',inplace=True)
weather_1.drop('Station',axis=1,inplace=True)
weather_2.drop('Station',axis=1,inplace=True)
#%%
train=add_weather_info_all(train,weather_1,weather_2)
test=add_weather_info_all(test,weather_1,weather_2)
#%%
add_date_features(test)
add_date_features(train)
#%%
train,test=add_dummies(train,test,['Species']) # block,street,trap
convert_categorical(train,test,['Block','Street','Trap']) #species
#%%
"""
Classifiers test
"""
ct=CatBoostClassifier()
rf=ensemble.RandomForestClassifier(n_estimators=1000)
gb=ensemble.GradientBoostingClassifier()
ad=ensemble.AdaBoostClassifier()
clfs=[rf,gb,ad,ct]
#cross_val_score(ad,train,y,scoring='roc_auc',cv=10).mean()
skf=StratifiedKFold(n_splits=3,random_state=42,shuffle=True)
for clf in clfs:
        aucs=[]
        for train_index,test_index in skf.split(train.values,y):
                X_train, X_test = train.values[train_index], train.values[test_index]
                y_train, y_test = y[train_index], y[test_index]
                clf.fit(X_train,y_train)
                preds=clf.predict(X_test)
                aucs.append(roc_auc_score(y_test,preds))
        print(np.array(aucs).mean())
#%%
"""
Selected GBC model
"""
clf=ensemble.GradientBoostingClassifier()
clf.fit(train,y)
preds=clf.predict_proba(test)[:,1]
sample['WnvPresent']=preds
sample.to_csv('sub.csv',index=False)
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
names = [train.columns[i] for i in indices]
plt.figure()
plt.title("Feature Importance")
plt.bar(range(train.shape[1]), importances[indices])
plt.xticks(range(train.shape[1]), names, rotation=90)
plt.show()
#%%
"""
Voting classifier
"""
est=[("rf",ensemble.RandomForestClassifier()),("xgb",ensemble.GradientBoostingClassifier()),("cat",CatBoostClassifier()),("ada",ensemble.AdaBoostClassifier())]
vt=ensemble.VotingClassifier(est,'soft',n_jobs=-1,weights=[1,5,3,2])
vt.fit(train,y)
preds=vt.predict_proba(test)[:,1]
sample['WnvPresent']=preds
sample.to_csv('subvt.csv',index=False)
#%%
"""
XGB parameter tuning
"""
from sklearn.model_selection import GridSearchCV
parameters = {
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "min_samples_split": np.linspace(0.1, 0.5, 12),
    "min_samples_leaf": np.linspace(0.1, 0.5, 12),
    "max_depth":[3,5,8],
    "max_features":["log2","sqrt"],
    "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    "n_estimators":[10]
    }
clf = GridSearchCV(ensemble.GradientBoostingClassifier(), parameters, cv=10, n_jobs=-1)
clf.fit(train, y)
print(clf.best_params_)
#%%
"""
Neural Network
"""
from keras.layers import Dense,Dropout,Input
from keras.models import Model
drpout_rate=0.5
input_layer=Input(shape=(train.shape[1],))
nn=Dense(int(train.shape[1]/2),activation='relu')(input_layer)
nn=Dropout(drpout_rate)(nn)
nn=Dense(100,activation='relu')(nn)
nn=Dropout(drpout_rate)(nn)
nn=Dense(1,activation='sigmoid')(nn)
model=Model(input_layer,nn)
model.compile(loss='binary_crossentropy',optimizer='adam')
model.fit(train,y,batch_size=40,epochs=500,verbose=0)
model.predict(test)
preds=model.predict(test)
sample['WnvPresent']=preds
sample.to_csv('subnn.csv',index=False)

