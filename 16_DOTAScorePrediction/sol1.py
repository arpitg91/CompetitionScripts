import pandas as pd
import numpy as np
from scipy import linalg
from numpy import dot
import os,sys

store={}
for filename in ['hero_data','train9','train1','test9','test1','sample_submission']:
    store[filename]=pd.read_csv('01.RawData/%s.csv'%filename)
    
a=pd.concat([store['train9'],store['test9']])
a['kda_ratio']=np.clip(a['kda_ratio'],2000,5500)
user_mean1=a.groupby(['user_id'])['kda_ratio'].mean()
hero_mean1=a.groupby(['hero_id'])['kda_ratio'].mean()
user_mean2=a['kda_ratio']-a['hero_id'].map(hero_mean1)
user_mean2=user_mean2.groupby(a['user_id']).mean()
hero_mean2=a['kda_ratio']-a['user_id'].map(user_mean1)
hero_mean2=hero_mean2.groupby(a['hero_id']).mean()

c=store['test1'].copy()
c['F1']=c['user_id'].map(user_mean1)
c['F2']=c['user_id'].map(user_mean2)
c['F3']=c['hero_id'].map(hero_mean1)
c['F4']=c['hero_id'].map(hero_mean2)
c['kda_ratio']=c['F1']+c['F4']
c['kda_ratio']=949.4+0.2383*c['F1']+0.8295*c['F2']+0.4924*c['F3']+0.6333*c['F4']
c[['id','kda_ratio']].to_csv('02.Submissions/4.UserHeroMean7.csv',index=False)