import pandas as pd
import sys,os
import numpy as np

os.chdir('C:/Users/arpit.goel/Documents/Projects/Kaggle/24_RecommendationDesign')
df_train=pd.read_csv('01.RawData/train.csv')
df_test=pd.read_csv('01.RawData/test.csv')
df_chal=pd.read_csv('01.RawData/challenge_data.csv')
df_sample=pd.read_csv('01.RawData/sample_submission_J0OjXLi_DDt3uQN.csv')
df_chal['publish_date']=pd.to_datetime(df_chal['publish_date'],format='%d-%m-%Y')
df_chal.sort_values(by='publish_date',inplace=True)
df_chal['challenge_id']=np.arange(len(df_chal))

users=df_train[['user_id']].drop_duplicates()
np.random.seed(1234)
users['dtype']=np.where(np.random.rand(len(users))>0.75,'valid','train')
users=users.set_index('user_id')['dtype']
df_train['ds']=df_train['user_id'].map(users)
df_test['ds']='test'
master=pd.merge(pd.concat([df_train,df_test]),df_chal,left_on=['challenge'],right_on=['challenge_ID'])
master.sort_values(by=['user_id','challenge_sequence'],inplace=True)

def get_datasegment(master,valid=True):
    if valid==True:
        a1=master[master['ds'].isin(['train'])]
        a2=master[master['ds'].isin(['valid'])]
        a3=a2[a2['challenge_sequence']<=10]
        a4=a2[a2['challenge_sequence']>10]
        return a1,a3,a4
    else:
        return master[master['ds'].isin(['train','valid'])],master[master['ds'].isin(['test'])],[]

train,test,target=get_datasegment(master,valid=False)

a1=pd.concat([train,test])
a1['sample']=a1['user_id']%10

freq_items=[]
group=a1[a1['sample']==1][['user_id','challenge_id','challenge_sequence']]
for name,group in a1.groupby('sample'):
    a2=pd.merge(group,group,on=['user_id'])
    a2=a2[a2['challenge_sequence_x']!=a2['challenge_sequence_y']]
    a2['wgt']=a2['challenge_sequence_x']-a2['challenge_sequence_y']
    a2['wgt']=np.where(a2['wgt']>0,0.7,0.8)**np.abs(a2['wgt'])
    summ=a2.groupby(['challenge_id_x','challenge_id_y'])['wgt'].sum().reset_index()
    freq_items.append(summ)
    
freq_items=pd.concat(freq_items).groupby(['challenge_id_x','challenge_id_y'])['wgt'].sum()
freq_items=freq_items.sort_values(ascending=False).to_frame('count_comb').reset_index()
freq_items=freq_items[freq_items['count_comb']>5]
print freq_items.shape

predicted_challenges={}
decay_factor=0.075
test['wgt']=test['challenge_sequence'].map(lambda x: decay_factor**(10-x))

i=0
for name,group in test[['user_id','challenge_id','challenge_sequence','wgt']].groupby(['user_id']):
    i+=1
    if i%1000==0:
        print i
    t1=pd.merge(group,freq_items,left_on=['challenge_id'],right_on=['challenge_id_x'])
    t1=t1[~t1['challenge_id_y'].isin(group['challenge_id'])]
    t1['count_comb']=t1['count_comb']*t1['wgt']
    predicted_challenges[name]=t1.groupby('challenge_id_y')['count_comb'].sum().sort_values(ascending=False).head(3).index.tolist()
    
output=[]
for key,value in predicted_challenges.items():
    output.append([str(int(key))+'_11',value[0] if len(value)>=1 else 1])
    output.append([str(int(key))+'_12',value[1] if len(value)>=2 else 2])
    output.append([str(int(key))+'_13',value[2] if len(value)>=3 else 3])
predicted_challenges_df=pd.DataFrame(output,columns=['user_sequence','challenge_id'])
predicted_challenges_df=pd.merge(predicted_challenges_df,df_chal[['challenge_id','challenge_ID']],on='challenge_id')
predicted_challenges_df['challenge']=predicted_challenges_df['challenge_ID']
predicted_challenges_df[['user_sequence','challenge']].to_csv('03.Submissions/04.FreqItemsets2_Decay0.7_0.8_0.075.csv',index=False)    