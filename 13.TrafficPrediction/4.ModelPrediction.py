
# coding: utf-8

# In[113]:

import os,sys
os.chdir('C:/Users/arpit.goel/Documents/Projects/Kaggle/13.TrafficPrediction')
from datetime import datetime
from datetime import date
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pyplot as plt # to visualize data
from pandas.tools.plotting import autocorrelation_plot # to visualize and configure the parameters of ARIMA model
from statsmodels.tsa.arima_model import ARIMA # to make an ARIMA model that fits the data

df_train=pd.read_csv('train_aWnotuB.csv',parse_dates=['DateTime'])
df_test=pd.read_csv('test_BdBKkAj.csv',parse_dates=['DateTime'])

monthly_regression_params={1: [2.93, 18.59], 2: [0.9, 6.18], 3: [0.5, 8.65], 4: [0.17, 5.19]}


# In[114]:

def get_vars(df1):
    df=df1.copy()
    df['year']=df['DateTime'].dt.year
    df['month']=df['DateTime'].dt.month
    df['day']=df['DateTime'].dt.day
    df['hour']=df['DateTime'].dt.hour
    df['day_of_week']=df['DateTime'].dt.dayofweek
    df['week_nbr']=df['DateTime'].dt.week
    df['flag_sunday']=(df['day_of_week']==6).astype(np.int64)
    df['flag_saturday']=(df['day_of_week']==5).astype(np.int64)
    df['flag_friday']=(df['day_of_week']==4).astype(np.int64)
    df['flag_monday']=(df['day_of_week']==0).astype(np.int64)
    df['flag_weekday']=(df['day_of_week']<=4).astype(np.int64)
    df['flag_junction_1']=(df['Junction']==1).astype(np.int64)
    df['flag_junction_2']=(df['Junction']==2).astype(np.int64)
    df['flag_junction_3']=(df['Junction']==3).astype(np.int64)
    df['flag_junction_4']=(df['Junction']==4).astype(np.int64)
    df['flag_last_day_of_month']=(df['day']==df['month'].map({1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31})).astype(np.int64)
    df['encoded_month']=12*df['year']+df['month']-24191
    df['date']=10000*df['year']+100*df['month']+df['day']
    df['encoded_date']=(df['DateTime']-date(2015,11,1)).dt.days
    slope=df['Junction'].map(lambda x: monthly_regression_params[x][0])
    intercept=df['Junction'].map(lambda x: monthly_regression_params[x][1])
    df['monthly_avg']=np.round(intercept+slope*df['encoded_month'])
    return df

train=get_vars(df_train)
train=train.sort_values(by='DateTime')


# In[115]:

params = {1:[7,1,1],2:[7,1,1],3:[4,1,0],4:[7,1,1]}

arima_prediction=[]
for key, group in train.groupby('Junction'):
    data = np.array(group.groupby('encoded_date')['Vehicles'].mean())
    result = None
    arima = ARIMA(data,params[key])
    result = arima.fit(disp=False)
    #print(result.params)
    pred = result.predict(2,len(data)-1,typ='levels')
    train_prediction=pd.Series([0,0]+list(pred),index=group['encoded_date'].drop_duplicates().sort_values())
    test_prediction=pd.Series(result.predict(len(data),len(data)+122,typ='levels'),index=map(int,range(608,731)))
    arima_prediction.append(train_prediction.append(test_prediction))

arima_daily_predictions=pd.DataFrame(arima_prediction,index=[1,2,3,4]).T.stack().to_dict()


# In[116]:

ins=train.head(int(0.95*len(train)))
oos=train.tail(len(train)-len(ins))
oot=get_vars(df_test)
ins['Vehicles_clean']=np.clip(ins['Vehicles'],np.percentile(ins['Vehicles'],0.01),np.percentile(ins['Vehicles'],0.99))
hourly_avg=ins.groupby(['hour','Junction'])['Vehicles_clean'].mean().unstack()
hourly_avg=(hourly_avg/hourly_avg.max()).stack()
hourly_avg=hourly_avg.map(lambda x: 'night' if x<0.70 else 'peak' if x>0.9 else 'regular').to_dict()

get_avgs=lambda x: ins.groupby(x)['Vehicles'].mean()

def get_other_vars(df):
    df['time_tag']=df.apply(lambda x: hourly_avg[x['hour'],x['Junction']],axis=1)
    df['flag_peak']=(df['time_tag']=='peak').astype(np.int64)
    df['flag_night']=(df['time_tag']=='night').astype(np.int64)
    df['flag_regular']=(df['time_tag']=='regular').astype(np.int64)
    for var,level in [['month_timetag',['month','time_tag']],['month_hour',['month','hour']],                     ['week_timetag',['week_nbr','time_tag']],['week_hour',['week_nbr','hour']],                     ['weekday_timetag',['day_of_week','time_tag']],['weekday_hour',['day_of_week','hour']],                     ['weekend_timetag',['flag_weekday','time_tag']],['weekend_hour',['flag_weekday','hour']],                     ['week',['week_nbr']],['month',['month']],['weekday',['day_of_week']]]:
        avgs=get_avgs(level).reset_index().rename(columns={'Vehicles':'avg_%s'%var})
        df=pd.merge(df,avgs,on=level,how='left')
    df['arima_prediction']=np.round(df.apply(lambda x: arima_daily_predictions[x['encoded_date'],x['Junction']],axis=1))
    return df
ins=get_other_vars(ins)
oos=get_other_vars(oos)
oot=get_other_vars(oot)


# In[118]:

print ins.columns
model_cols=[u'Junction',u'year', u'month', u'day',u'hour', u'day_of_week', u'week_nbr']+            [x for x in ins.columns if 'flag' in x]+            [x for x in ins.columns if 'avg' in x]

model_cols=['monthly_avg','avg_weekday_hour','week_nbr','avg_week_hour','day','avg_week_timetag','day_of_week','Junction','hour','year']

T_train_xgb = xgb.DMatrix(ins[model_cols], ins['Vehicles'])

xgtrain = xgb.DMatrix(ins[model_cols], ins['Vehicles'])
xgvalid = xgb.DMatrix(oos[model_cols], oos['Vehicles'])
xgtest = xgb.DMatrix(oot[model_cols])
watchlist  = [(xgvalid,'eval'), (xgtrain,'train')]

param = {'max_depth':3, 'eta':0.01, 'silent':1, 'objective':'reg:linear', 'min_child_weight':4,         'nthread':3, 'gamma': 0.0,'subsample':0.75, 'colsample_bytree':0.9, 'learning_rate':0.1,'reg_alpha':0.1}
print param
param['eval_metric'] = 'rmse'
plst = param.items()
evallist = [(xgvalid,'oos'), (xgtrain,'ins')]
model = xgb.train(plst, xgtrain, 200, evallist)
ins_pred_actual  = model.predict(xgtrain)
oos_pred_actual = model.predict(xgvalid)
oot_pred_actual = model.predict(xgtest)


# In[63]:

param_test1 = {
 'reg_alpha':[0.05, 0.1, 0.5]
          }
gsearch1 = GridSearchCV(estimator = xgb.sklearn.XGBRegressor( learning_rate =0.01, n_estimators=100, max_depth=3,
 min_child_weight=4, gamma=0, subsample=0.75, colsample_bytree=0.9,
 objective= 'reg:linear', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, n_jobs=1,iid=False, cv=5)
gsearch1.fit(ins[model_cols], ins['Vehicles'])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


# In[119]:

oot['Vehicles']=np.round(model.predict(xgb.DMatrix(oot[model_cols])))
oot[['ID','Vehicles']].to_csv('XGBoost1.csv',index=False)

ins['score']=np.round(model.predict(xgb.DMatrix(ins[model_cols])))
ins['error']=np.abs(ins['score']-ins['Vehicles'])
ins['error_rank']=np.round(ins['error'].rank(pct=True),3)
ins.to_csv('chk1.csv')


# In[117]:

imp_days=ins.groupby(['Junction','encoded_date','month','day'])['Vehicles'].mean().reset_index()
imp_days_prev=imp_days.drop(['month','day'],axis=1)
imp_days_prev['encoded_date']=imp_days['encoded_date']+1
imp_days=pd.merge(imp_days,imp_days_prev,on=['Junction','encoded_date'])
imp_days['Delta']=100*np.abs(imp_days['Vehicles_x']/imp_days['Vehicles_y']-1)
print imp_days['Delta'].describe(percentiles=[0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99])
imp_days=imp_days[imp_days['Delta']>25]
imp_days=imp_days.groupby(['month','day'])['Delta'].count()
imp_days=imp_days[imp_days>3].reset_index()
imp_days['flag_imp_day']=1

#ins=pd.merge(ins,imp_days,on=['month','day'])


# In[121]:

ins.groupby(['day','month'])['score'].mean().unstack().fillna(0).to_clipboard()

