import pandas as pd
import os,sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime,date,timedelta
from itertools import product
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

os.chdir('C:/Users/arpit.goel/Documents/Projects/Kaggle/15.RecruitRestarauntPrediction')
store=pd.HDFStore('01.RawData/DataStore.h5')
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
df=df[df['visit_date']>=date(2016,12,31)]
df=df[['reserve_visitors','type','visit_date','reserve_date']].reset_index()
df=np.log1p(df.groupby(['air_store_id','visit_date','reserve_date','type'])['reserve_visitors'].sum()).reset_index()
df=pd.merge(df,store['store_info'],left_on=['air_store_id'],right_index=True,how='left')
reservation_groups=[]
reservation_groups.append(df.groupby(['cluster','visit_date','reserve_date','type'])['reserve_visitors'].sum().unstack())
reservation_groups.append(df.groupby(['cluster','cuisine','visit_date','reserve_date','type'])['reserve_visitors'].sum().unstack())
reservation_groups.append(df.groupby(['cluster','area_type','visit_date','reserve_date','type'])['reserve_visitors'].sum().unstack())
reservation_groups.append(df.groupby(['cluster','cuisine','area_type','visit_date','reserve_date','type'])['reserve_visitors'].sum().unstack())
reservation_groups.append(df.groupby(['air_store_id','cluster','cuisine','area_type','visit_date','reserve_date','type'])['reserve_visitors'].sum().unstack())
df2=store['store_info'].reset_index()
df2=df2[df2['air_store_id'].str[:3]=='air']
for i in range(len(reservation_groups)):
    reservation_groups[i].columns=['reserve_%d_%s'%(i,x) for x in reservation_groups[i].columns]
    reservation_groups[i]=reservation_groups[i].reset_index()
    keys=reservation_groups[i].reset_index().columns.intersection(df2.columns).tolist()
    df2=pd.merge(df2,reservation_groups[i],on=keys,how='left')
store['reserve_master']=df2

# Read sample submission
df=pd.read_csv('01.RawData/sample_submission.csv')
df['air_store_id']=df['id'].map(lambda x: '_'.join(x.split('_')[:2]))
df['visit_date']=pd.to_datetime(df['id'].map(lambda x: x.split('_')[2]))
df.set_index(['air_store_id','visit_date'],inplace=True)
store['sample_submission']=df

#Get visits master data
df=pd.merge(np.log1p(store['air_visit_data']).reset_index(),store['store_info'],left_on=['air_store_id'],right_index=True)
df=df[df['visit_date']<date(2016,6,1)]
visits_groups=[]
visits_groups.append(df.groupby(['cluster','visit_date'])['visitors'].sum().to_frame('visits_0').reset_index())
visits_groups.append(df.groupby(['cluster','cuisine','visit_date'])['visitors'].sum().to_frame('visits_1').reset_index())
visits_groups.append(df.groupby(['cluster','area_type','visit_date'])['visitors'].sum().to_frame('visits_2').reset_index())
visits_groups.append(df.groupby(['cluster','area_type','cuisine','visit_date'])['visitors'].sum().to_frame('visits_3').reset_index())
visits_groups.append(df.groupby(['air_store_id','visit_date'])['visitors'].sum().to_frame('visits_4').reset_index())
df2=store['store_info'].reset_index()
df2=df2[df2['air_store_id'].str[:3]=='air']
for i in range(len(reservation_groups)):
    keys=visits_groups[i].reset_index().columns.intersection(df2.columns).tolist()
    df2=pd.merge(df2,visits_groups[i],on=keys,how='left')
store['visit_master']=df2[['air_store_id','visit_date','visits_0','visits_1','visits_2','visits_3','visits_4']]

# store.close()    
# store={}
# for file in ['air_reserve','air_store_info','air_visit_data','date_info','hpg_reserve','hpg_store_info','store_id_relation','sample_submission']:
    # store[file]=pd.read_hdf('01.RawData/DataStore.h5', file)

visits=np.log1p(pd.concat([store['air_visit_data'],store['sample_submission']])['visitors']).unstack()
min_dates=visits.stack().reset_index().groupby(['air_store_id'])['visit_date'].min()
min_dates.name='first_date'

def get_timespan(df,dt,window,interval=1):
    start=dt-timedelta(days=window*interval-7)
    return df[pd.date_range(start,periods=window,freq='%dD'%interval)]

def prepare_dataset(dt):
    features={}
    for i in [1,2,3,7,14,21,35,42,56,91,147]:
        dcut=get_timespan(visits,dt,i)
        features['avg_visitors_%d'%i]=dcut.mean(axis=1).values
        features['min_visitors_%d'%i]=dcut.min(axis=1).values
        features['max_visitors_%d'%i]=dcut.max(axis=1).values
        features['med_visitors_%d'%i]=dcut.median(axis=1).values
        features['cnt_visitors_%d'%i]=dcut.count(axis=1).values
    for i,j in product(range(7),[1,2,3,5,8,10,12,13,21]):
        dcut=get_timespan(visits,dt-timedelta(days=i),j,7)
        features['avg_visitors_wd%d_%d'%(i,j)]=dcut.mean(axis=1).values
        features['min_visitors_wd%d_%d'%(i,j)]=dcut.min(axis=1).values
        features['max_visitors_wd%d_%d'%(i,j)]=dcut.max(axis=1).values
        features['med_visitors_wd%d_%d'%(i,j)]=dcut.median(axis=1).values
        features['cnt_visitors_wd%d_%d'%(i,j)]=dcut.count(axis=1).values
        
    visits1=store['visit_master']
    visits1=visits1[visits1['visit_date']>=dt+timedelta(-365)]
    visits1=visits1[visits1['visit_date']<=dt+timedelta(40-365)]
    for j in range(5):
        visits2=visits1.set_index(['air_store_id','visit_date'])['visits_%d'%j].unstack().reindex(visits.index)
        for i in range(39):
            features['visits_365_%d_%d'%(j,i)]=visits2.iloc[:,i]
            features['visits_364_%d_%d'%(j,i)]=visits2.iloc[:,i+1]
        
    reserve=store['reserve_master']
    reserve=reserve[reserve['reserve_date']<dt]
    reserve=reserve[reserve['visit_date']>=dt]
    reserve=reserve[reserve['visit_date']<=dt+timedelta(39)]
    for j,k in product(range(5),['air','hpg']):
        reserve1=reserve.groupby(['air_store_id','visit_date'])['reserve_%d_%s'%(j,k)].sum().unstack().reindex(visits.index)
        for i in range(39):
            if dt+timedelta(days=i) in reserve1.columns:
                features['reserve_%d_%s_%d'%(j,k,i)]=reserve1[dt+timedelta(days=i)].values
    X=pd.DataFrame(features,index=visits.index)
    if (dt-date(2017,4,22)).days<0:
        y=visits[pd.date_range(dt+timedelta(days=1),periods=39)].values
        y=pd.DataFrame(y,index=visits.index)
        return X,y
    return X

def prepare_data_single_model():
    baseline_pred_vars=['avg_visitors_wd_21','med_visitors_wd_21','cnt_visitors_147','avg_visitors_7','med_visitors_wd_13',\
                    'cnt_visitors_wd_13','avg_visitors_wd_10','avg_visitors_3','min_visitors_wd_21','min_visitors_wd_5',\
                    'med_visitors_wd_10','min_visitors_wd_1','avg_visitors_wd_13','cnt_visitors_wd_21','min_visitors_3',\
                    'med_visitors_3','cnt_visitors_7','max_visitors_147','med_visitors_7','med_visitors_wd_8','avg_visitors_wd_2',\
                    'min_visitors_21','avg_visitors_wd_8','min_visitors_91','max_visitors_56','min_visitors_7','avg_visitors_21',\
                    'max_visitors_42','cnt_visitors_91']
    other_vars=['reserve_%d_%s'%(x,y) for x,y in product(range(5),['air','hpg'])]
    other_vars+=['visits_%d_%d'%(x,y) for x,y in product([364,365],range(5))]
    output=[[],[],[],[],[]]
    for i in range(39):
        ins_msk=store['train_y'].iloc[:,i].fillna(0)>0
        oos_msk=store['val_y'].iloc[:,i].fillna(0)>0
        idv=['%s_visitors_%d'%(x,y) for x,y in product(['min','max','avg','med','cnt'],[1,2,3,7,14,21,35,42,56,91,147])]
        idv+=['%s_visitors_wd%d_%d'%(x,6-i%7,y) for x,y in product(['min','max','avg','med','cnt'],[1,2,3,5,8,10,13,21])]
        idv+=['reserve_%d_%s_%d'%(x,y,i) for x,y in product(range(5),['air','hpg'])]
        idv+=['visits_%d_%d_%d'%(x,y,i) for x,y in product([364,365],range(5))]
        idv_labels=['%s_visitors_%d'%(x,y) for x,y in product(['min','max','avg','med','cnt'],[1,2,3,7,14,21,35,42,56,91,147])]
        idv_labels+=['%s_visitors_wd_%d'%(x,y) for x,y in product(['min','max','avg','med','cnt'],[1,2,3,5,8,10,13,21])]
        idv_labels+=['reserve_%d_%s'%(x,y) for x,y in product(range(5),['air','hpg'])]
        idv_labels+=['visits_%d_%d'%(x,y) for x,y in product([364,365],range(5))]
        dfs=[]
        dfs.append(store['train_x'].loc[ins_msk,idv].rename(columns=dict(zip(idv,idv_labels)))[baseline_pred_vars+other_vars])
        dfs.append(store['val_x'].loc[oos_msk,idv].rename(columns=dict(zip(idv,idv_labels)))[baseline_pred_vars+other_vars])
        dfs.append(store['test_x'].loc[:,idv].rename(columns=dict(zip(idv,idv_labels)))[baseline_pred_vars+other_vars])
        dfs.append(store['train_y'].loc[ins_msk,i].to_frame('tgt'))
        dfs.append(store['val_y'].loc[oos_msk,i].to_frame('tgt'))
        for j in range(5):
            dfs[j]['day']=i
            output[j].append(dfs[j])
    output=[pd.concat(x) for x in output]
    return output

    
data=[prepare_dataset(i.date()) for i in pd.date_range(date(2016,12,31),periods=11,freq='7D')]
X,Y=zip(*data)
store['train_x']=pd.concat(list(X)[:-1])
store['train_y']=pd.concat(list(Y)[:-1])
store['val_x']=X[-1]
store['val_y']=Y[-1]
store['test_x']=prepare_dataset(date(2017,4,22))

data=prepare_data_single_model()
for x,y in zip(['train1_x','val1_x','test1_x','train1_y','val1_y'],data):
    store[x]=y
