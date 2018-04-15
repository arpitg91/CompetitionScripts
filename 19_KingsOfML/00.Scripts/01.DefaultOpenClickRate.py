import pandas as pd
import numpy as np
import os,sys
os.chdir('C:/Users/arpit.goel/Documents/Projects/Kaggle/25_KingsOfML')

df_train=pd.read_csv('01.RawData/train.csv')
df_test=pd.read_csv('01.RawData/test_BDIfz5B.csv')

open_rates=df_train.groupby(['user_id'])['is_open'].mean()
click_rates=df_train.groupby(['user_id'])['is_click'].mean()

df_test['is_click']=df_test['user_id'].map(open_rates).fillna(0)
df_test[['id','is_click']].to_csv('03.Submission/01.AvgOpenRate.csv',index=False)
df_test['is_click']=df_test['user_id'].map(click_rates).fillna(0)
df_test[['id','is_click']].to_csv('03.Submission/02.AvgClickRate.csv',index=False)