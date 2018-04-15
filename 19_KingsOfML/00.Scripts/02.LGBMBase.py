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
df_train['communication_type']=df_train['campaign_id'].map(comm_type)
signals_1=df_train.groupby(['user_id','month'])[['rcv','is_click','is_open']].sum().add_prefix('sum_')
signals_1=signals_1.stack().unstack(1).fillna(0).cumsum(axis=1).stack().unstack(1).reset_index()
signals_1['month']=signals_1['month']+1
signals_2=df_train.groupby(['user_id','communication_type','month'])[['rcv','is_click','is_open']].sum().add_prefix('sum_')
signals_2=signals_2.stack().unstack(2).fillna(0).cumsum(axis=1).stack().unstack([2,1]).fillna(0)
signals_2.columns=signals_2.columns.get_level_values(0)+'_'+signals_2.columns.get_level_values(1).astype(np.str)
signals_2=signals_2.reset_index()
signals_2['month']=signals_2['month']+1
signals=pd.merge(signals_1,signals_2,on=['user_id','month'])

df_train['source']='train'
df_train['month']=df_train['send_date'].dt.month
df_test['source']='test'
df_test['month']=13
master=pd.concat([df_train,df_test])
master=pd.merge(master,signals,on=['user_id','month'],how='left').fillna(0)
master['campaign_rank']=master.groupby('user_id')['campaign_id'].rank()

sig_sets=[x for x in master.columns if 'sum_rcv' in x]
for sig in sig_sets:
    master[sig.replace('rcv','open_rate')]=1.0*master[sig.replace('rcv','is_open')]/(master[sig.replace('rcv','rcv')]+0.0001)
    master[sig.replace('rcv','click_rate')]=1.0*master[sig.replace('rcv','is_click')]/(master[sig.replace('rcv','rcv')]+0.0001)
    master[sig.replace('rcv','click_per_open')]=1.0*master[sig.replace('rcv','is_click')]/(master[sig.replace('rcv','is_open')]+0.0001)

#Add campaign signals from campaign contents
def get_free_text_vars(df,var,label):
    df['sig_ft_0_%s'%label]=df[var].astype(np.str).str.len()                                                             # Num characters
    df['sig_ft_2_%s'%label]=df[var].astype(np.str).map(lambda x: re.sub('[^#$%&*:;<=>?@\^_`|~]','',x)).str.len()         # Num Special Characters
    df['sig_ft_3_%s'%label]=df[var].astype(np.str).map(lambda x: re.sub('[^0-9]','',x)).str.len()                        # Num numerals
    df['sig_ft_4_%s'%label]=df[var].astype(np.str).map(lambda x: re.sub('[^a-z]','',x)).str.len()                        # Num lower case
    df['sig_ft_5_%s'%label]=df[var].astype(np.str).map(lambda x: re.sub('[^A-Z]','',x)).str.len()                        # Num upper case
    df['sig_ft_6_%s'%label]=df[var].astype(np.str).map(lambda x: re.sub('[^ ]','',x)).str.len()                          # Num spaces
    return df

cmpgn_sigs=df_cmpgn.set_index(['campaign_id'])
cmpgn_sigs=get_free_text_vars(cmpgn_sigs,'subject','sub')
cmpgn_sigs=get_free_text_vars(cmpgn_sigs,'email_body','email')
cmpgn_sigs.rename(columns={'total_links':'sig_links1','no_of_internal_links':'sig_links2',
        'no_of_images':'sig_images','no_of_sections':'sig_sections'},inplace=True)
cmpgn_sigs1=cmpgn_sigs[[x for x in cmpgn_sigs.columns if 'sig' in x]]

#Add campaign level signals from campaign population
cmpgn_sigs=master[['sum_rcv','sum_is_open','sum_is_click','campaign_id']]
cmpgn_sigs['rcv']=1
cmpgn_sigs['sig_cmpgn_rcv_0']=1
cmpgn_sigs['sig_cmpgn_rcv_1']=(cmpgn_sigs['sum_rcv']>=1).astype(np.int64)
cmpgn_sigs['sig_cmpgn_rcv_2']=(cmpgn_sigs['sum_rcv']>=2).astype(np.int64)
cmpgn_sigs['sig_cmpgn_open_1']=(cmpgn_sigs['sum_is_open']>=1).astype(np.int64)
cmpgn_sigs['sig_cmpgn_open_2']=(cmpgn_sigs['sum_is_open']>=2).astype(np.int64)
cmpgn_sigs['sig_cmpgn_click_1']=(cmpgn_sigs['sum_is_click']>=1).astype(np.int64)
cmpgn_sigs['sig_cmpgn_click_2']=(cmpgn_sigs['sum_is_click']>=2).astype(np.int64)

vars1=['sig_cmpgn_%s_%d'%(x,y) for x in ['rcv','open','click'] for y in [1,2]]
cmpgn_sigs=cmpgn_sigs.groupby('campaign_id')[['rcv','sig_cmpgn_rcv_0']+vars1].sum()
for var in vars1:
    cmpgn_sigs[var.replace('cmpgn_','cmpgn_rate_')]=1.0*cmpgn_sigs[var]/cmpgn_sigs['rcv']
cmpgn_sigs2=cmpgn_sigs[[x for x in cmpgn_sigs.columns if 'sig' in x]]
cmpgn_sigs=pd.concat([cmpgn_sigs1,cmpgn_sigs2],axis=1)

#Add campaign signals to master
master=pd.merge(master,cmpgn_sigs,left_on=['campaign_id'],right_index=True)
    
def get_train_test_data(valid=True):
    signals=[x for x in master.columns if x[:3] in ['sum','com','sig'] or x in ['campaign_rank']]
    train_months=[8,9] if valid==True else [8,9,10,11,12]
    valid_months=[10,11,12] if valid==True else [13]
    train_X=master[master.month.isin(train_months)][signals]
    valid_X=master[master.month.isin(valid_months)][signals]
    train_Y=master[master.month.isin(train_months)][['is_click','is_open']]
    valid_Y=master[master.month.isin(valid_months)][['is_click','is_open']]
    return train_X,valid_X,train_Y,valid_Y
        
master.set_index(['id','user_id','campaign_id'],inplace=True)
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

# params = {
        # 'learning_rate': 0.05,
        # 'max_depth':3,
        # 'boosting_type': 'gbdt',
        # 'objective': 'binary',
        # 'metric': 'auc',
        # 'num_leaves': 2**3-1,
        # 'verbose': -1,
        # 'data_random_seed': 1,
        # 'bagging_fraction': 0.7,
        # 'bagging_freq': 5,
        # 'feature_fraction': 0.6,
        # 'min_data_in_leaf': 250,
    # }

# lgb_train = lgb.Dataset(train_X, train_Y['is_open'])
# watchlist = [lgb_train]
# gbm = lgb.train(params,lgb_train,num_boost_round=25,valid_sets=watchlist,verbose_eval=5)

# valid_X['is_click']=gbm.predict(valid_X)
# valid_X[['is_click']].reset_index().to_csv('03.Submission/04.LGBMOpen.csv',index=False)

