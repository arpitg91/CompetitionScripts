import pandas as pd
import numpy as np
import os,sys
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

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

#Get campaign type and campaign rank
comm_type=df_cmpgn.set_index('campaign_id')['communication_type']
comm_type=comm_type.map(lambda x: 'Upcoming Events' if x in ['Others','Webinar'] else x)
comm_type=comm_type.astype('category').cat.codes

campaign_dates=pd.concat([df_train[['campaign_id','send_date']],df_test[['campaign_id','send_date']]])
campaign_dates=campaign_dates.groupby(['campaign_id'])['send_date'].min().to_frame('first_send_date').reset_index()
campaign_dates['campaign_type']=campaign_dates['campaign_id'].map(comm_type)
campaign_dates['campaign_rank1']=campaign_dates['first_send_date'].rank()
campaign_dates['campaign_rank2']=campaign_dates.groupby(['campaign_type'])['campaign_rank1'].rank()
campaign_dates=campaign_dates[['campaign_id','campaign_type','campaign_rank1','campaign_rank2']]
campaign_dates.index=campaign_dates.campaign_id

#Training Data Signals
signal_raw=df_train.copy()
signal_raw['rcv']=1
signal_raw=pd.merge(signal_raw,campaign_dates,on='campaign_id')

#User Signals
signals_1=signal_raw.set_index(['user_id','campaign_rank1'])[['rcv','is_open','is_click']]
signals_1=signals_1.stack().unstack(1).fillna(0).cumsum(axis=1)
cols1=signals_1.columns
signals_1=signals_1[cols1[:-1]]
signals_1.columns=cols1[1:]
signals_1=signals_1.stack().unstack(1).fillna(0)
signals_1.columns=['sig_rcv','sig_open','sig_click']

#User-Campaign Signals
signals_2=signal_raw.set_index(['user_id','campaign_type','campaign_rank2'])[['rcv','is_open','is_click']]
signals_2=signals_2.stack().unstack(2).fillna(0).cumsum(axis=1)
cols1=signals_2.columns
signals_2=signals_2[cols1[:-1]]
signals_2.columns=cols1[1:]
signals_2=signals_2.stack().unstack(2).fillna(0)
signals_2.columns=['sig_rcv_type','sig_open_type','sig_click_type']

signal_master=signal_raw[['id','user_id','campaign_id','campaign_rank1','campaign_rank2','is_click','campaign_type']]
signal_master=pd.merge(signal_master,signals_1.reset_index(),on=['user_id','campaign_rank1'],how='left')
signal_master=pd.merge(signal_master,signals_2.reset_index(),on=['user_id','campaign_type','campaign_rank2'],how='left')
signal_master=signal_master.fillna(0).set_index(['user_id','campaign_id','id'])
signal_master=pd.concat([signal_master,pd.get_dummies(signal_master['campaign_type'],prefix='sig_cmpgn_type')],axis=1)
signal_train=signal_master.drop(labels=['campaign_rank1','campaign_rank2','campaign_type'],axis=1)

#Test Data Signals

#User Signals
signals_1=signal_raw.groupby(['user_id'])[['rcv','is_open','is_click']].sum()
signals_1.columns=['sig_rcv','sig_open','sig_click']

#User-Campaign Signals
signals_2=signal_raw.groupby(['user_id','campaign_type'])[['rcv','is_open','is_click']].sum()
signals_2.columns=['sig_rcv_type','sig_open_type','sig_click_type']

signal_master=df_test[['id','user_id','campaign_id']]
signal_master['campaign_type']=signal_master['campaign_id'].map(campaign_dates['campaign_type'])
signal_master=pd.merge(signal_master,signals_1.reset_index(),on=['user_id'],how='left')
signal_master=pd.merge(signal_master,signals_2.reset_index(),on=['user_id','campaign_type'],how='left')
signal_master=signal_master.fillna(0).set_index(['user_id','campaign_id','id'])
signal_master=pd.concat([signal_master,pd.get_dummies(signal_master['campaign_type'],prefix='sig_cmpgn_type')],axis=1)
signal_test=signal_master.drop(labels=['campaign_type'],axis=1)

#Master Signal Set
signal_master=pd.concat([signal_train,signal_test]).fillna(0)

#Add Historic Open Rate, Click Rate and clicks per open
for sigset in ['','_type']:
    signal_master['sig_click_rate%s'%sigset]=1.0*signal_master['sig_click%s'%sigset]/(signal_master['sig_rcv%s'%sigset]+0.0001)
    signal_master['sig_open_rate%s'%sigset]=1.0*signal_master['sig_open%s'%sigset]/(signal_master['sig_rcv%s'%sigset]+0.0001)
    signal_master['sig_click_per_open%s'%sigset]=1.0*signal_master['sig_click%s'%sigset]/(signal_master['sig_open%s'%sigset]+0.0001)

#Add campaign signals from campaign contents
import re,string
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
cmpgn_sigs=signal_master[['sig_rcv','sig_open','sig_click']].reset_index()
cmpgn_sigs['rcv']=1
cmpgn_sigs['sig_cmpgn_rcv_0']=1
cmpgn_sigs['sig_cmpgn_rcv_1']=(cmpgn_sigs['sig_rcv']>=1).astype(np.int64)
cmpgn_sigs['sig_cmpgn_rcv_2']=(cmpgn_sigs['sig_rcv']>=2).astype(np.int64)
cmpgn_sigs['sig_cmpgn_open_1']=(cmpgn_sigs['sig_open']>=1).astype(np.int64)
cmpgn_sigs['sig_cmpgn_open_2']=(cmpgn_sigs['sig_open']>=2).astype(np.int64)
cmpgn_sigs['sig_cmpgn_click_1']=(cmpgn_sigs['sig_click']>=1).astype(np.int64)
cmpgn_sigs['sig_cmpgn_click_2']=(cmpgn_sigs['sig_click']>=2).astype(np.int64)

vars1=['sig_cmpgn_%s_%d'%(x,y) for x in ['rcv','open','click'] for y in [1,2]]
cmpgn_sigs=cmpgn_sigs.groupby('campaign_id')[['rcv','sig_cmpgn_rcv_0']+vars1].sum()
for var in vars1:
    cmpgn_sigs[var.replace('cmpgn_','cmpgn_rate_')]=1.0*cmpgn_sigs[var]/cmpgn_sigs['rcv']
cmpgn_sigs2=cmpgn_sigs[[x for x in cmpgn_sigs.columns if 'sig' in x]]
cmpgn_sigs=pd.concat([cmpgn_sigs1,cmpgn_sigs2],axis=1)

#Add campaign signals to master
# signal_master1=pd.merge(signal_master.reset_index(),cmpgn_sigs2,left_on=['campaign_id'],right_index=True)
# signal_master1.set_index(['user_id','campaign_id','id'],inplace=True)
signal_master1=signal_master

#Get datasets for modelling. Turn the below parameter to False while subitting.
validation=False

train_campaigns=range(29,47) if validation==True else range(29,55)
valid_campaigns=range(47,55) if validation==True else range(55,81)

train=signal_master1[signal_master1.index.get_level_values(1).isin(train_campaigns)]
valid=signal_master1[signal_master1.index.get_level_values(1).isin(valid_campaigns)]

signals=[x for x in signal_master1.columns if 'sig' in x]
train_X=train[signals]
train_Y=train['is_click']
valid_X=valid[signals]
valid_Y=valid['is_click']
print [x.shape for x in [train_X,train_Y,valid_X,valid_Y]]

lgb_train = lgb.Dataset(train_X, train_Y)
if validation==True:
    lgb_test = lgb.Dataset(valid_X, valid_Y)
    watchlist = [lgb_train,lgb_test]
else:
    watchlist = [lgb_train]
    
for param in [1]:
    params = {
        'learning_rate': 0.001,
        'max_depth':5,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 2**4-1,
        'verbose': -1,
        'data_random_seed': 1,
        'bagging_fraction': 0.85, 
        'bagging_freq': 2,
        'feature_fraction': 0.6,
        'min_data_in_leaf': 100,
    }
    
    gbm = lgb.train(params,lgb_train,num_boost_round=50,valid_sets=watchlist,verbose_eval=10)
    print (param,roc_auc_score(train_Y, gbm.predict(train_X)))
    # print (param,roc_auc_score(valid_Y, gbm.predict(valid_X)))
    
valid_X['is_click']=gbm.predict(valid_X)
valid_X.reset_index()[['id','is_click']].to_csv('03.Submission/05.LGBMCampaign.csv',index=False)
