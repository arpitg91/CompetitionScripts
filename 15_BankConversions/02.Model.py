import pandas as pd
import numpy as np
import os,sys
import matplotlib.pyplot as plt
from datetime import datetime

def prepare_data(df1):
    df=df1.copy()
    df['DOB1']=df['DOB'].map(lambda x: datetime(1900+int(x[6:]),int(x[3:5]),int(x[:2])) if len(str(x))>=8 else datetime(1997,1,1))
    df['Lead_Creation_Date1']=df['Lead_Creation_Date'].map(lambda x: datetime(2000+int(x[6:]),int(x[3:5]),int(x[:2])) if len(str(x))>=8 else datetime(1997,1,1))
    # for i in range(12):
        # df['FEAT_ID_%d'%i],_=pd.factorize(df.index.map(lambda x: x[3+i]))
    df['FEAT_GENDER'],_=pd.factorize(df['Gender'])
    df['FEAT_DOB_YEAR'],_=pd.factorize(df['DOB1'].dt.year)
    df['FEAT_DOB_MONTH'],_=pd.factorize(df['DOB1'].dt.month)
    df['FEAT_LCD_MONTH'],_=pd.factorize(df['Lead_Creation_Date1'].dt.month)
    df['FEAT_LCD_WEEK'],_=pd.factorize(df['Lead_Creation_Date1'].dt.week)
    df['FEAT_LCD_DOW'],_=pd.factorize(df['Lead_Creation_Date1'].dt.dayofweek)
    df['FEAT_DOB_MONTH'],_=pd.factorize(df['DOB1'].dt.month)
    df['FEAT_DOB_LCD_MONTH_DIFF'],_=pd.factorize(df['Lead_Creation_Date1'].dt.month-df['DOB1'].dt.month)
    df['FEAT_DOB_LCD_WEEK_DIFF'],_=pd.factorize(np.clip(df.Lead_Creation_Date1.dt.week-df.DOB1.dt.week,-10,10))
    df['FEAT_DOB_LCD_DAY_DIFF'],_=pd.factorize(np.clip(df.Lead_Creation_Date1.dt.dayofyear-df.DOB1.dt.dayofyear,-10,10))

    for i in range(6):
        df['FEAT_CITY1_%d'%i],_=pd.factorize(df['City_Code'].map(lambda x: str(x)[:i+1]))
    df['FEAT_CITY2'],_=pd.factorize(df['City_Category'])
    for i in range(4):
        df['FEAT_EMP1_%d'%i],_=pd.factorize(df['Employer_Code'].map(lambda x: str(x)[:-i-1]))
    df['FEAT_EMP2'],_=pd.factorize(df['Employer_Category1'])
    df['FEAT_EMP3'],_=pd.factorize(df['Employer_Category2'])
    df['FEAT_INCOME1']=np.clip(np.log1p(df['Monthly_Income']),0,10).astype(np.int64)
    df['CONT_FEAT_INCOME1']=np.clip(df['Monthly_Income'].fillna(-99999),-99999,9500)
    df['FEAT_INCOME2']=df['Monthly_Income'].astype(np.int64)%10
    df['FEAT_BANK1'],_=pd.factorize(df['Customer_Existing_Primary_Bank_Code'])
    df['FEAT_BANK2'],_=pd.factorize(df['Customer_Existing_Primary_Bank_Code'].map(lambda x: str(x)[:-1]))
    df['FEAT_BANK3'],_=pd.factorize(df['Primary_Bank_Type'])
    df['FEAT_CONTACT1'],_=pd.factorize(df['Contacted'])
    df['FEAT_CONTACT2'],_=pd.factorize(df['Source'])
    df['FEAT_CONTACT3'],_=pd.factorize(df['Source'].map(lambda x: x[:3]))
    df['FEAT_CONTACT4'],_=pd.factorize(df['Source'].map(lambda x: x[3]))
    df['FEAT_CONTACT5'],_=pd.factorize(df['Source_Category'])
    df['FEAT_EMI1'],_=pd.factorize(df['Existing_EMI']==0)
    df['FEAT_EMI2'],_=pd.factorize(pd.cut(df['Existing_EMI'],[0,100,300,600,1200,2400,3600,10000000]))
    df['FEAT_LOAN1'],_=pd.factorize(df['Loan_Amount'].isnull())
    df['FEAT_LOAN2'],_=pd.factorize(pd.cut(df['Loan_Amount'].fillna(-1),[-100,-1,0,5000,10000,20000,30000,50000,100000,10000000]))
    df['CONT_FEAT_LOAN1']=np.clip(df['Loan_Amount'].fillna(-99999),-99999,100000)
    df['CONT_FEAT_LOAN2']=df[['Loan_Amount','Loan_Period','Existing_EMI']].apply(lambda x: x[0]/(x[1]*x[2]) if x[1]>0 and x[2]>0 else -99999,axis=1).fillna(-99999)
    df['CONT_FEAT_LOAN3']=df['Interest_Rate'].fillna(-99999)
    df['CONT_FEAT_EMI1']=df['Existing_EMI'].fillna(-99999)
    df['FEAT_LOAN3'],_=pd.factorize(df['Loan_Period'].fillna(-1))
    df['FEAT_LOAN4'],_=pd.factorize((df['Interest_Rate'].fillna(-10)/5).astype(np.int64))
    df['FEAT_VAR1'],_=pd.factorize(df['Var1'])
    for col in [x for x in df.columns if x[:4] in ('FEAT')]:
        a=df[col].value_counts()
        df[col.replace('FEAT','COUNT')]=df[col].map(a)
        if a.min()<=100:
            a=a[a>100]
            a1=df[col].astype('category')
            a2=df[col].map(a).fillna(-99999).astype('category')
            df[col]=a1
            df[col+'_CLEAN']=a2
        else:
            df[col]=df[col].astype('category')
    return df[[x for x in df.columns if x[:4] in ('FEAT','Appr','CONT','COUN')]]

os.chdir('C:/Users/arpit.goel/Documents/Projects/Kaggle/18_BankConversions')
df_train=pd.read_csv('01.RawData/train.csv',index_col=['ID'])
df_test=pd.read_csv('01.RawData/test.csv',index_col=['ID'])
master=prepare_data(pd.concat([df_train,df_test]))

psis=[]
for col in [x for x in master.columns if x[:4] in ('FEAT')]:
    a=pd.crosstab(master[col],master['Approved'].notnull())
    a=np.clip(a,1,100000)
    a=a/a.sum()
    a1=a.iloc[:,0]
    a2=a.iloc[:,1]
    psis.append([col,((a2-a1)*np.log(a2/a1)).fillna(0).sum()])
psis=pd.DataFrame(psis,columns=['var','psi'])
psis=psis[psis['psi']<=0.01]
vars=psis['var'].tolist()+[x for x in master.columns if x[:4] in ('CONT','COUN')]
train=master[master['Approved'].notnull()][vars+['Approved']]
test=master[master['Approved'].isnull()][vars]

idv=[x for x in train.columns if x[:4] in ('FEAT','CONT','COUN')]
msk=np.random.rand(len(train))<0.95
ins=train[msk]
oos=train[~msk]

if 1==1:
    from catboost import CatBoostClassifier
    model=CatBoostClassifier(iterations=1000, learning_rate=0.0005, depth=5, loss_function='Logloss')
    cat_cols=ins[idv].dtypes
    cat_cols.index=range(len(idv))
    cat_cols=cat_cols[cat_cols=='category'].index.tolist()
    model.fit(ins[idv],ins['Approved'],cat_cols)
    preds_proba = model.predict(ins[idv], prediction_type="Probability")
    ins['Approved1']=model.predict(ins[idv], prediction_type="Probability")[:,1]
    oos['Approved1']=model.predict(oos[idv], prediction_type="Probability")[:,1]
    test['Approved1']=model.predict(test[idv], prediction_type="Probability")[:,1]
    test.reset_index()[['ID','Approved']].to_csv('04.Submissions/01.Depth5CatBoost.csv',index=False)

if 1==1:
    import lightgbm as lgb
    params = {'objective': 'binary','metric': 'auc','boosting': 'gbdt','max_depth': 5,'num_leaves': 2**5,
              'learning_rate': 0.0005,'num_rounds': 500,
        'bagging_fraction': 0.8,'feature_fraction': 0.8,
        'bagging_freq': 1,'max_bin': 100}

    train_lgb = lgb.Dataset(ins[idv], ins['Approved'])
    model = lgb.train(params, train_lgb)

    ins['Approved2']=model.predict(ins[idv])
    oos['Approved2']=model.predict(oos[idv])
    test['Approved2']=model.predict(test[idv])
    # test.reset_index()[['ID','Approved']].to_csv('04.Submissions/03.LightGBM.csv',index=False)

test['Approved']=test[['Approved1','Approved2']].mean(axis=1)
test.reset_index()[['ID','Approved']].to_csv('04.Submissions/04.CatBoostLightGBMEnsemble.csv',index=False)
