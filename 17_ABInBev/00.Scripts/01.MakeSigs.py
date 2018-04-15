import pandas as pd
import numpy as np
import os,sys
from datetime import date

os.chdir('C:/Users/arpit.goel/Documents/Projects/Kaggle/23_ABInBev')

historical_volume=pd.read_csv('01.RawData/historical_volume.csv')
ymk=historical_volume[['YearMonth']].drop_duplicates().sort_values(by=['YearMonth'])
ymk['YearMonthKey']=ymk['YearMonth'].map(lambda x: 12*np.floor(x/100-2013)+x%100)
ymk['YearMonthDt']=ymk['YearMonth'].map(lambda x: date(int(x/100),int(x%100),1))
ymk=ymk.set_index('YearMonth')
historical_volume['YearMonthKey']=historical_volume['YearMonth'].map(ymk['YearMonthKey'])

def get_ts_df(df,start,interval,period=1):
    return df.iloc[:,start-interval-1:start-1:period]

def get_historical_volume_sigs(start_time=60):
    sigs=[]
    for i in [1,2,3,4,12]:
        t1=get_ts_df(df1,start_time,i)
        sigs.append(t1.mean(axis=1).to_frame('vol_avg_ty_p%02dm'%i))
        sigs.append(t1.std(axis=1).to_frame('vol_std_ty_p%02dm'%i))
        sigs.append(t1.min(axis=1).to_frame('vol_min_ty_p%02dm'%i))
        sigs.append(t1.max(axis=1).to_frame('vol_max_ty_p%02dm'%i))
        sigs.append(t1.skew(axis=1).to_frame('vol_skew_ty_p%02dm'%i))
        
        t1=get_ts_df(df1,start_time-12,i)
        sigs.append(t1.mean(axis=1).to_frame('vol_avg_ly_p%02dm'%i))
        sigs.append(t1.std(axis=1).to_frame('vol_std_ly_p%02dm'%i))
        sigs.append(t1.min(axis=1).to_frame('vol_min_ly_p%02dm'%i))
        sigs.append(t1.max(axis=1).to_frame('vol_max_ly_p%02dm'%i))
        sigs.append(t1.skew(axis=1).to_frame('vol_skew_ly_p%02dm'%i))
        
    sigs.append(get_ts_df(df1,start_time,12,12).mean(axis=1).to_frame('vol_avg_sm_ly'))
    if start_time<61:
        sigs.append(df1.iloc[:,start_time-1].to_frame('tgt'))
    sigs=pd.concat(sigs,axis=1)
    sigs['vol_avg_ty_p03m_p04m']=sigs['vol_avg_ty_p03m']/(sigs['vol_avg_ty_p04m']+0.001)
    sigs['vol_avg_ty_p03m_p12m']=sigs['vol_avg_ty_p03m']/(sigs['vol_avg_ty_p12m']+0.001)
    sigs['vol_avg_ty_p04m_p12m']=sigs['vol_avg_ty_p04m']/(sigs['vol_avg_ty_p12m']+0.001)
    for i in [1,2,3,4,12]:
        sigs['vol_avg_diff_ty_ly_p%02dm'%i]=sigs['vol_avg_ty_p%02dm'%i]-sigs['vol_avg_ly_p%02dm'%i]
        sigs['vol_avg_diff_ly_p%02dm'%i]=sigs['vol_avg_sm_ly']-sigs['vol_avg_ly_p%02dm'%i]
    sigs['YearMonthKey']=start_time
    return sigs.set_index('YearMonthKey',append=True)
    
df1=historical_volume.set_index(['Agency','SKU','YearMonthKey'])['Volume'].unstack()
master=pd.concat([get_historical_volume_sigs(month) for month in range(25,62)]).reset_index()
master['MonthKey']=master['YearMonthKey'].map(lambda x: (x-1)%12+1)

demographics=pd.read_csv('01.RawData/demographics.csv')
master=pd.merge(master,demographics,on=['Agency'],how='left')

def get_other_sigs(df,value):
    output=[]
    for month in range(25,62):
        sigs=[]
        sigs.append(get_ts_df(df,month,12,12).mean(axis=1).to_frame('%s_sm_ly'%(value)))
        for i in [1,2,3,4,12]:
            sigs.append(get_ts_df(df,month,i).mean(axis=1).to_frame('%s_ty_p%02dm'%(value,i)))
            sigs.append(get_ts_df(df,month-12,i).mean(axis=1).to_frame('%s_ly_p%02dm'%(value,i)))
        sigs=pd.concat(sigs,axis=1)
        sigs['%s_ty_p03m_p04m'%value]=sigs['%s_ty_p03m'%value]/(sigs['%s_ty_p04m'%value]+0.001)
        sigs['%s_ty_p03m_p12m'%value]=sigs['%s_ty_p03m'%value]/(sigs['%s_ty_p12m'%value]+0.001)
        sigs['%s_ty_p04m_p12m'%value]=sigs['%s_ty_p04m'%value]/(sigs['%s_ty_p12m'%value]+0.001)
        for i in [1,2,3,4,12]:
            sigs['%s_avg_diff_ty_ly_p%02dm'%(value,i)]=sigs['%s_ty_p%02dm'%(value,i)]-sigs['%s_ly_p%02dm'%(value,i)]
            sigs['%s_avg_diff_ly_p%02dm'%(value,i)]=sigs['%s_sm_ly'%(value)]-sigs['%s_ly_p%02dm'%(value,i)]
        sigs['YearMonthKey']=month
        output.append(sigs)
    output=pd.concat(output).reset_index()  
    return output

for a,b in [('industry_soda_sales','soda'),('industry_volume','indus'),('price_sales_promotion','promo')]:
    df=pd.read_csv('01.RawData/%s.csv'%a)
    df['YearMonthKey']=df['YearMonth'].map(ymk['YearMonthKey'])
    if b!='promo':
        df=df.drop(labels=['YearMonth'],axis=1).set_index(['YearMonthKey']).T
        df=get_other_sigs(df,b)
        master=pd.merge(master,df,on=['YearMonthKey'],how='left')
    else:
        df2=df.drop(labels=['YearMonth'],axis=1).set_index(['Agency','SKU','YearMonthKey'])['Sales'].unstack()
        df2=get_other_sigs(df2,'sp')
        master=pd.merge(master,df2,on=['Agency','SKU','YearMonthKey'],how='left')
        df2=df.drop(labels=['YearMonth'],axis=1).set_index(['Agency','SKU','YearMonthKey'])['Promotions'].unstack()
        df2=get_other_sigs(df2,'promo')
        master=pd.merge(master,df2,on=['Agency','SKU','YearMonthKey'],how='left')

df=pd.read_csv('01.RawData/weather.csv')
df['MonthKey']=df['YearMonth'].map(lambda x: x%100)
df=df.groupby(['MonthKey'])['Avg_Max_Temp'].mean().to_frame('avg_temp').reset_index()
master=pd.merge(master,df,on=['MonthKey'],how='left')

df=pd.read_csv('01.RawData/event_calendar.csv')
df['YearMonthKey']=df['YearMonth'].map(ymk['YearMonthKey'])
df1=df[df['YearMonth']==201701]
df1['YearMonthKey']=61
df=pd.concat([df,df1])
master=pd.merge(master,df,on=['YearMonthKey'],how='left')

predictors=list(master.columns.difference(['Agency','SKU','YearMonthKey','tgt','MonthKey','YearMonth','index_x','index_y']))
import lightgbm as lgb
master1=master[(master['tgt']>10)&(master['YearMonthKey']>34)]
master1['tgt']=np.clip(master1['tgt'],0,10000)
ins=master1[master1['YearMonthKey']<58]
oos=master1[master1['YearMonthKey'].isin([58,59,60])]
oot=master[master['YearMonthKey']==61]
def get_accuracy(col,tgt):
    err=np.abs(col-tgt).sum()
    return 1-err/tgt.sum()


params = {
        'learning_rate': 0.01,
        'max_depth':7,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 2**5-1,
        'verbose': -1,
        'data_random_seed': 1,
        'bagging_fraction': 0.9,
        'bagging_freq': 5,
        'feature_fraction': 0.55,
        'min_data_in_leaf': 25,
    }
    

lgb_train = lgb.Dataset(ins[predictors], ins['tgt'])
lgb_test = lgb.Dataset(oos[predictors], oos['tgt'])
watchlist = [lgb_train,lgb_test]
gbm = lgb.train(params,lgb_train,num_boost_round=1000,valid_sets=watchlist,verbose_eval=100)
print (get_accuracy(gbm.predict(oos[predictors]),oos['tgt']))
oot['prediction1']=gbm.predict(oot[predictors])

master1=master[master['MonthKey']==1]
master1=master1[(master1['tgt']>10)]
master1['tgt']=np.clip(master1['tgt'],0,10000)
ins=master1[master1['YearMonthKey'].isin([25,37,49])]
params = {
        'learning_rate': 0.01,
        'max_depth':7,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 2**5-1,
        'verbose': -1,
        'data_random_seed': 1,
        'bagging_fraction': 0.9,
        'bagging_freq': 5,
        'feature_fraction': 0.55,
        'min_data_in_leaf': 25,
    }
    

lgb_train = lgb.Dataset(ins[predictors], ins['tgt'])
lgb_test = lgb.Dataset(oos[predictors], oos['tgt'])
watchlist = [lgb_train,lgb_test]
gbm = lgb.train(params,lgb_train,num_boost_round=1000,valid_sets=watchlist,verbose_eval=100)
print (get_accuracy(gbm.predict(oos[predictors]),oos['tgt']))
oot['prediction2']=gbm.predict(oot[predictors])
oot['prediction']=np.clip(oot[['prediction1','prediction2']].mean(axis=1),0,100000)

sample_submission=pd.read_csv('01.RawData/SampleSubmission/volume_forecast.csv')
sample_submission=pd.merge(sample_submission,oot[['Agency','SKU','prediction']],on=['Agency','SKU'],how='left')
sample_submission['Volume']=sample_submission['prediction'].fillna(0)
sample_submission[['Agency','SKU','Volume']].to_csv('03.Submissions/1.LGBMFirst.csv',index=False)
sample_submission[['Agency','SKU','Volume']].to_csv('03.Submissions/volume_forecast.csv',index=False)