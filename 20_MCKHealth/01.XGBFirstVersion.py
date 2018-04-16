import pandas as pd
import numpy as np

# Read the input datasets
df_train=pd.read_csv('01.RawData/train_ajEneEa.csv')
df_test=pd.read_csv('01.RawData/test_v2akXPA.csv')
df_subm=pd.read_csv('01.RawData/sample_submission_1.csv')

# Create features
def get_signals(df1):
    df=df1.copy()
    df['sig_id1']=df['id']
    df['sig_id2']=1.0*df.index/len(df)
    gender_sigs=pd.get_dummies(df['gender']).add_prefix('sig_gender_')
    work_sigs=pd.get_dummies(df['work_type']).add_prefix('sig_work_')
    smoking_sigs=pd.get_dummies(df['smoking_status'].fillna('missing')).add_prefix('sig_smoke_')
    df=pd.concat([df,gender_sigs,work_sigs,smoking_sigs],axis=1)
    for sig in ['age','hypertension','heart_disease','avg_glucose_level','bmi']:
        df['sig_%s'%sig]=df[sig].fillna(-9999)
    df['sig_married']=(df['ever_married']=='Yes').astype(np.int8)
    df['sig_residence']=(df['Residence_type']=='Urban').astype(np.int8)
    df['sig_bmi_missing']=df['bmi'].isnull().astype(np.int8)
    output_sigs=[x for x in df.columns if x[:3] in ['sig','str']]
    return df[output_sigs]

sigs_train=get_signals(df_train)
sigs_test=get_signals(df_test)

# Performance deteriorated. So removed the variables
# # Add prediction of BMI using input variables
# bmi_master=pd.concat([sigs_train,sigs_test])
# bmi_train=bmi_master[bmi_master['sig_bmi_missing']==0]
# bmi_idv=[x for x in bmi_train.columns if x[:3] in ['sig'] and 'bmi' not in x]

# # Train an XGBoost to predict BMI
# from xgboost.sklearn import XGBRegressor
# xgb = XGBRegressor(n_estimators=50, learning_rate=0.2, gamma=0, subsample=1,
#                                colsample_bytree=1, max_depth=3,min_child_weight=1)
# xgb.fit(bmi_train[bmi_idv], bmi_train['sig_bmi'].values,eval_metric='rmse')
# sigs_train['sig_bmi_pred']=xgb.predict(sigs_train[bmi_idv])
# sigs_test['sig_bmi_pred']=xgb.predict(sigs_test[bmi_idv])

# Make training set, target
idv=[x for x in sigs_train.columns if x[:3] in ['sig']]
X_ins=sigs_train[idv]
X_oos=sigs_test[idv]
Y_ins=sigs_train['stroke']

#Train XGBoost for stroke probability
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
clf1 = XGBClassifier( learning_rate =0.01, n_estimators=500, max_depth=3, min_child_weight=1, gamma=0.0, subsample=0.7,
 colsample_bytree=0.45, reg_alpha=0.1, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27)
clf1.fit(X_ins, Y_ins,eval_metric='auc')

# Performance deteriorated. So, keeping all the variables instead of selecting top ones
# feat_imp=pd.Series(xgb.booster().get_fscore()).sort_values(ascending=False)
# idv=feat_imp[feat_imp.cumsum()/feat_imp.sum()<0.95].index.tolist()
# xgb.fit(X_ins[idv], Y_ins,eval_metric='auc')

#Train Random Forest
from sklearn.ensemble import RandomForestClassifier
clf2=RandomForestClassifier(n_estimators=50,max_depth=5)
clf2.fit(X_ins, Y_ins)

#Train Logistic Regression
from sklearn.linear_model import LogisticRegression
clf3=LogisticRegression(C=0.1)
clf3.fit(X_ins, Y_ins)

#Train KNN
from sklearn.neighbors import KNeighborsClassifier
clf4=KNeighborsClassifier(n_neighbors=500)
clf4.fit(X_ins, Y_ins)

#Train Extra Tree Classifier
from  sklearn.ensemble import ExtraTreesClassifier
clf5=ExtraTreesClassifier(n_estimators=30, max_depth=4,random_state=1234)
clf5.fit(X_ins, Y_ins)

pred=[]
for clf in [clf1,clf2,clf5]:
    pred.append(clf.predict_proba(X_oos)[:,1])

#Output the final file
df_test['stroke']= (np.array(pred).T*np.array([ 1,0,0])).sum(axis=1)
df_test[['id','stroke']].to_csv('02.Submission/07.XGB_withIndex.csv',index=False)
