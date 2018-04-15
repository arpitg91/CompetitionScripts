def apk(actual, predicted, k=3):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    
    actual = list(actual)
    predicted = list(predicted)
    
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)
            
    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=3):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])
    
    
import pandas as pd
import sys,os
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

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

train,test,target=get_datasegment(master,valid=True)

challenge_counts=train['challenge_id'].value_counts().head(13)
test_challenges=test.groupby(['user_id'])['challenge_id'].apply(lambda x: set(x)).to_frame('challenges')
predicted_challenges={}
for name,row in test_challenges.iterrows():
    predictions=challenge_counts.index.difference(row['challenges'])
    predicted_challenges[name]=challenge_counts[predictions].sort_values(ascending=False).head(3).values

prediction=pd.Series(predicted_challenges)
actual=target.groupby(['user_id'])['challenge_id'].apply(lambda x: list(x))
mapk(actual,prediction)    