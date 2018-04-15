import pandas as pd
import numpy as np
%matplotlib inline 
import os,sys
os.chdir('C:/Users/arpit.goel/Documents/Projects/Kaggle/19_DOTA')

store={}
for filename in ['hero_data','train9','train1','test9','test1','sample_submission']:
    store[filename]=pd.read_csv('01.RawData/%s.csv'%filename)
hero_data=pd.read_csv('01.RawData/hero_data.csv',index_col=['hero_id'])
# primary_attr - strength, agility, intelligence
# str > int, int> agi, agi > str

hero_data=pd.read_csv('01.RawData/hero_data.csv',index_col=['hero_id'])
for k,v in hero_data[hero_data.columns[3:]].apply(lambda x: x.unique().shape[0]).to_dict().items():
    if v>1:
        hero_data['FEAT_%s'%k]=hero_data['%s'%k]

for role in set([y for x in hero_data['roles'].str.split(':') for y in x]):
    hero_data['FEAT_role_%s'%role]=hero_data['roles'].map(lambda x: role in x).astype(np.int8)
    
cat_features=pd.get_dummies(hero_data[['primary_attr','attack_type']],prefix='FEAT')
hero_features=pd.concat([hero_data,cat_features],axis=1)
hero_features.columns=hero_features.columns.str.upper()
hero_features=hero_features.iloc[:,hero_features.columns.str[:4]=='FEAT']

means=store['test9'].groupby(['user_id'])['kda_ratio'].mean()
submission=store['test1'].copy()
submission['kda_ratio']=submission['user_id'].map(means)
submission[['id','kda_ratio']].to_csv('02.Submissions/2.UserMeans1.csv',index=False)
submission['kda_ratio']=submission['user_id'].map(means)
submission['kda_ratio']=submission['kda_ratio']-submission['kda_ratio'].mean()+3545.967206
submission[['id','kda_ratio']].to_csv('02.Submissions/2.UserMeans2.csv',index=False)
submission['kda_ratio']=submission['user_id'].map(means)
submission['kda_ratio']=(submission['kda_ratio']-submission['kda_ratio'].mean())/submission['kda_ratio'].std()
submission['kda_ratio']=submission['kda_ratio']*946.1+3545.967206
submission[['id','kda_ratio']].to_csv('02.Submissions/2.UserMeans3.csv',index=False)


