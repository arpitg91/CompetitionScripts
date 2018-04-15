import pandas as pd
import numpy as np
import os,sys,re,string
from sklearn.feature_extraction.text import CountVectorizer
import re,string

os.chdir('C:/Users/arpit.goel/Documents/Projects/Kaggle/25_KingsOfML')

# df_train=pd.read_csv('01.RawData/train.csv')
# df_test=pd.read_csv('01.RawData/test_BDIfz5B.csv')
# df_cmpgn=pd.read_csv('01.RawData/campaign_data.csv')
# df_sample=pd.read_csv('01.RawData/sample_submission_4fcZwvQ.csv')
# df_train['send_date']=pd.to_datetime(df_train['send_date'],format='%d-%m-%Y %H:%M')
# df_test['send_date']=pd.to_datetime(df_test['send_date'],format='%d-%m-%Y %H:%M')
# df_train.sort_values(by=['user_id','send_date'],inplace=True)
# df_test.sort_values(by=['user_id','send_date'],inplace=True)
# store=pd.HDFStore('01.RawData/DataStore.h5')
# store['train']=df_train
# store['test']=df_test
# store['cmpgn']=df_cmpgn
# store['sample']=df_sample
# store.close()

df_train=pd.read_hdf('01.RawData/DataStore.h5', 'train')
df_test=pd.read_hdf('01.RawData/DataStore.h5', 'test')
df_cmpgn=pd.read_hdf('01.RawData/DataStore.h5', 'cmpgn')
df_sample=pd.read_hdf('01.RawData/DataStore.h5', 'sample')
comm_type=df_cmpgn.set_index('campaign_id')['communication_type'].astype('category').cat.codes

df_train['month']=df_train.send_date.dt.month
df_train['rcv']=1
df_train.rename(columns={'is_click':'click','is_open':'open'},inplace=True)
signals_1=df_train.groupby(['user_id','month'])[['rcv','click','open']].sum().add_prefix('ever_')
signals_1=signals_1.stack().unstack(1).fillna(0).cumsum(axis=1).stack().unstack(1).reset_index()
signals_1['month']=signals_1['month']+1
signals_2=df_train.groupby(['user_id','month'])[['rcv','click','open']].sum().add_prefix('last_').reset_index()
signals_2['month']=signals_2['month']+1
signals=pd.merge(signals_1,signals_2,on=['user_id','month'],how='left')

df_train['source']='train'
df_train['month']=df_train['send_date'].dt.month
df_test['source']='test'
df_test['month']=13
master=pd.concat([df_train,df_test])
master=pd.merge(master,signals,on=['user_id','month'],how='left').fillna(0)
master['communication_type']=master['campaign_id'].map(comm_type)
master.set_index(['id','user_id','campaign_id'],inplace=True)

sig_sets=['ever','last']
for sig in sig_sets:
    master['%s_click_rate'%sig]     =1.0*master['%s_click'%sig]/(master['%s_rcv'%sig ]+0.0001)
    master['%s_open_rate'%sig]      =1.0*master['%s_open'%sig ]/(master['%s_rcv'%sig ]+0.0001)
    master['%s_click_per_open'%sig] =1.0*master['%s_click'%sig]/(master['%s_open'%sig]+0.0001)
    
def get_train_test_data(valid=True):
    signals=[x for x in master.columns if x[:4] in ['ever','last','comm']]
    train_months=[8,9] if valid==True else [8,9,10,11,12]
    valid_months=[10,11,12] if valid==True else [13]
    train_X=master[master.month.isin(train_months)][signals]
    valid_X=master[master.month.isin(valid_months)][signals]
    train_Y=master[master.month.isin(train_months)]['click']
    valid_Y=master[master.month.isin(valid_months)]['click']
    return train_X,valid_X,train_Y,valid_Y

train_X,valid_X,train_Y,valid_Y=get_train_test_data(valid=False)

from sklearn.metrics import roc_auc_score
import lightgbm as lgb

params = {
        'learning_rate': 0.001,
        'max_depth':4,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 2**3-1,
        'verbose': -1,
        'data_random_seed': 1,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'feature_fraction': 0.6,
        'min_data_in_leaf': 250,
    }

lgb_train = lgb.Dataset(train_X, train_Y['is_click'])
watchlist = [lgb_train]
gbm = lgb.train(params,lgb_train,num_boost_round=25,valid_sets=watchlist,verbose_eval=5)

valid_X['is_click']=gbm.predict(valid_X)
valid_X[['is_click']].reset_index().to_csv('03.Submission/05.LGBMCampaign.csv',index=False)

