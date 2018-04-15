import pandas as pd
import os,sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime,date,timedelta
from itertools import product
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge,Lasso

os.chdir('C:/Users/arpit.goel/Documents/Projects/Kaggle/15.RecruitRestarauntPrediction')
store={}
store['air_visit_data']=pd.read_csv('01.RawData/air_visit_data.csv',parse_dates=['visit_date'],index_col=['air_store_id','visit_date'])
store['date_info']=pd.read_csv('01.RawData/date_info.csv',parse_dates=['calendar_date'],index_col=['calendar_date'])
store['store_id_relation']=pd.read_csv('01.RawData/store_id_relation.csv')

#Get store info master data
df=pd.read_csv('01.RawData/hpg_store_info.csv',index_col=['hpg_store_id'])
df=pd.merge(df, store['store_id_relation'], how='left', left_index=True,right_on=['hpg_store_id'])
df.index=df['air_store_id'].fillna(df['hpg_store_id'])
df2=pd.read_csv('01.RawData/air_store_info.csv',index_col=['air_store_id'])
cuisine_mapping=pd.read_csv('01.RawData/cuisine_mapping.csv',index_col=['cuisine'])
cuisine_mapping=cuisine_mapping['mapping'].str.lower()
store_info=pd.concat([df2,df])
store_info['cuisine']=store_info['air_genre_name'].fillna(store_info['hpg_genre_name']).map(cuisine_mapping)
store_info['area_name']=store_info['air_area_name'].fillna(store_info['hpg_area_name'])
store_info['flag_prefecture']=store_info['area_name'].map(lambda x: '-to' in x or '-do' in x or '-fu' in x or '-ken' in x or 'Prefecture' in x or '-gun' in x).astype(np.int8)
store_info['flag_city']=store_info['area_name'].map(lambda x: '-shi' in x).astype(np.int8)
store_info['area_type']=store_info['flag_prefecture'].astype(np.str)+store_info['flag_city'].astype(np.str)
store_info[[x for x in store_info.columns if 'flag' in x]].apply(lambda x: '-'.join(map(str,x)),axis=1).value_counts()
store_info['cluster']= KMeans(n_clusters=10, random_state=0).fit(store_info[['latitude','longitude']]).labels_
store['store_info']=store_info[['cuisine','area_type','cluster']].groupby(level=0).first()

#Get reservation master data
df=pd.read_csv('01.RawData/air_reserve.csv',parse_dates=['visit_datetime','reserve_datetime'],index_col=['air_store_id'])
df['type']='air'
df2=pd.read_csv('01.RawData/hpg_reserve.csv',parse_dates=['visit_datetime','reserve_datetime'],index_col=['hpg_store_id'])
df2=pd.merge(df2, store['store_id_relation'], how='left', left_index=True,right_on=['hpg_store_id'])
df2.index=df2['air_store_id'].fillna(df2['hpg_store_id'])
df2['type']='hpg'
df=pd.concat([df,df2])
df['visit_date']=df['visit_datetime'].dt.date
df['reserve_date']=df['reserve_datetime'].dt.date
df=df[['reserve_visitors','type','visit_date','reserve_date']].reset_index()
df=np.log1p(df.groupby(['air_store_id','visit_date','reserve_date','type'])['reserve_visitors'].sum()).reset_index()
df=pd.merge(df,store['store_info'],left_on=['air_store_id'],right_index=True,how='left')
store['reserve_master']=df

cal=store['date_info'].reset_index()
cal['week_start']=cal['calendar_date']-pd.to_timedelta((cal['calendar_date'].dt.weekday+1)%7,unit='D')
cal=cal[~cal['day_of_week'].isin(['Saturday','Sunday'])]
cal=cal.groupby(['week_start'])['holiday_flg'].max().to_frame('num_holidays')

def get_ts_values(ds,period,label,reverse=False):
    weekly_tgts=[]
    for i,j in enumerate(ds.columns.get_level_values(-1)):
        if reverse==True:
            b=ds.iloc[:,i:min(i+period,ds.shape[1])]
        else: 
            b=ds.iloc[:,max(i-period,0):i]
        b.columns=['%s_%d'%(label,x) for x in range(b.shape[1])]
        b.loc[:,'week_start']=j
        weekly_tgts.append(b.set_index('week_start',append=True))
    return pd.concat(weekly_tgts)

def get_reserve_vars(pdate,group_level,c):
    c1=c[(c['reserve_week']<pdate)&(c['visit_week']>=pdate)]
    c2=c1.groupby(group_level+['visit_week'])['hpg'].sum().unstack()
    c2.columns=['hpg_reserve_%d'%((x-pdate).days/7) for x in c2.columns]
    c2['week_start']=pdate
    reserve1=c2.set_index('week_start',append=True)
    c2=c1.groupby(group_level+['visit_week'])['air'].sum().unstack()
    c2.columns=['air_reserve_%d'%((x-pdate).days/7) for x in c2.columns]
    c2['week_start']=pdate
    reserve2=c2.set_index('week_start',append=True)
    return reserve1,reserve2
    
def get_group_model(group_level):
    a=pd.merge(store['air_visit_data'].reset_index(),store['reserve_master'].reset_index(),on='air_store_id')
    a['week_start']=a['visit_date']-pd.to_timedelta((c['visit_date'].dt.weekday+1)%7,unit='D')
    weekly_visits=a[a['week_start']>=date(2016,6,1)]
    weekly_visits=weekly_visits.groupby(group_level+['week_start'])['visitors'].sum().to_frame('weekly_visits').unstack()

    tgts=get_ts_values(weekly_visits,6,'tgt_visitors',reverse=True)
    pvar=get_ts_values(weekly_visits,6,'prev_visits')
    cal_vars=get_ts_values(cal.T,6,'num_holidays',reverse=True).fillna(0)
    cal_vars.index=cal_vars.index.get_level_values(1)

    c=store['reserve_master'].copy()
    c['reserve_date']=c['reserve_date'].astype(np.datetime64)
    c['visit_date']=c['visit_date'].astype(np.datetime64)
    c['reserve_week']=c['reserve_date']-pd.to_timedelta((c['reserve_date'].dt.weekday+1)%7,unit='D')
    c['visit_week']=c['visit_date']-pd.to_timedelta((c['visit_date'].dt.weekday+1)%7,unit='D')
    c['interval']=(c['visit_week']-c['reserve_week']).dt.days/7
    c=c[(c['interval']>0)&(c['interval']<=6)]
    c=c.groupby(group_level+['reserve_week','visit_week','interval','type'])['reserve_visitors'].sum().unstack().reset_index()

    data=[get_reserve_vars(pdate,group_level,c) for pdate in pd.concat([tgts,pvar]).index.get_level_values(1).drop_duplicates()]
    reserve1,reserve2=zip(*data)
    reserve1=pd.concat(reserve1)
    reserve2=pd.concat(reserve2)

    master=pd.concat([tgts,pvar,reserve1,reserve2,cal_vars.reindex(tgts.index, level=1)],axis=1)
    test_vars1=weekly_visits.iloc[:,-6:]
    test_vars1.columns=['prev_visits_%d'%(5-i) for i in range(6)]
    test_reserve_1,test_reserve_2=get_reserve_vars(datetime(2017,4,23))
    test_reserve_1.index=test_vars1.index
    test_reserve_2.index=test_vars1.index
    test_data=pd.concat([test_vars1,test_reserve_1,test_reserve_2],axis=1).fillna(0)

    for i in range(6):
        test_data['num_holidays_%d'%i]=cal.loc[date(2017,4,23)+timedelta(days=7*i)].values[0]
        idv=[x for x in master.columns if x[:4]=='prev']
        idv+=['hpg_reserve_%d'%i,'air_reserve_%d'%i,'num_holidays_%d'%i]
        target='tgt_visitors_%d'%i
        train=master[idv+[target]].dropna(subset=[x for x in master.columns if x[:4]=='prev']+[target]).fillna(0)
        train=train[train.index.get_level_values(1)>=datetime(2017,1,1)]
        model=Lasso(alpha=0.01)
        model.fit(train[idv],train[target])
        print model.score(train[idv],train[target]),model.intercept_, model.coef_
        test_data[target]=model.predict(test_data[idv].fillna(0))
    
    return test_data
    
# get_group_model(['cluster']).to_csv('03.Profile/03.WeeklyPredictions_Cluster.csv')
