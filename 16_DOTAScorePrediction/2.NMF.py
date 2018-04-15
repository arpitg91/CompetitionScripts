import pandas as pd
import numpy as np
from scipy import linalg
from numpy import dot
import os,sys
os.chdir('C:/Users/arpit.goel/Documents/Projects/Kaggle/19_DOTA')

#Read all the input files
store={}
for filename in ['hero_data','train9','train1','test9','test1','sample_submission']:
    store[filename]=pd.read_csv('01.RawData/%s.csv'%filename)
    store[filename]['source']=filename

# Define function for matrix decomposition. If 0 as matrix entity, do not fit that value
def nmf(mat, latent_features, max_iter=100, error_limit=1e-6, fit_error_limit=1e-6):
    X=mat.fillna(0).values
    test=store['train1']
    # Define index for prediction
    pred_index=mat.fillna(0).stack().to_frame('kda_ratio').reset_index()
    pred_index=pred_index['user_id'].astype(str)+'_'+pred_index['hero_id'].astype(str)

    
    eps = 1e-5
    print 'Starting NMF decomposition with {} latent features and {} iterations.'.format(latent_features, max_iter)
    # mask
    mask = np.sign(X)

    # initial matrices. A is random [0,1] and Y is A\X.
    rows, columns = X.shape
    np.random.seed(1234)
    A = np.random.rand(rows, latent_features)
    A = np.maximum(A, eps)

    Y = linalg.lstsq(A, X)[0]
    Y = np.maximum(Y, eps)

    masked_X = mask * X
    X_est_prev = dot(A, Y)
    for i in range(1, max_iter + 1):
        # ===== updates =====
        # Matlab: A=A.*(((W.*X)*Y')./((W.*(A*Y))*Y'));
        top = dot(masked_X, Y.T)
        bottom = (dot((mask * dot(A, Y)), Y.T)) + eps
        A *= top / bottom

        A = np.maximum(A, eps)
        # print 'A',  np.round(A, 2)

        # Matlab: Y=Y.*((A'*(W.*X))./(A'*(W.*(A*Y))));
        top = dot(A.T, masked_X)
        bottom = dot(A.T, mask * dot(A, Y)) + eps
        Y *= top / bottom
        Y = np.maximum(Y, eps)
        # print 'Y', np.round(Y, 2)


        # ==== evaluation ====
        if i % 1 == 0 or i == 1 or i == max_iter:
            print 'Iteration {}:'.format(i),
            X_est = dot(A, Y)
            err = mask * (X_est_prev - X_est)
            fit_residual = np.sqrt(np.sum(err ** 2))
            X_est_prev = X_est

            curRes = linalg.norm(mask * (X - X_est), ord='fro')
            print 'fit residual', np.round(fit_residual, 4),
            print 'total residual', np.round(curRes, 4),
            if curRes < error_limit or fit_residual < fit_error_limit:
                break
            # Validation error to find optimal number of iterations.
            pred=pd.DataFrame(A.dot(Y)).stack().to_frame('kda_ratio')
            pred.index=pred_index
            test['prediction']=test['id'].map(pred['kda_ratio'])
            test['error']=np.square(test['prediction']-test['kda_ratio'])
            print 'test error', np.sqrt(test['error'].mean())
    return A,Y

mat=pd.concat([store['test9'],store['train9'],store['train1']]).groupby(['user_id','hero_id'])
# Remove outliers
mat=np.clip(mat['kda_ratio'].mean(),2000,6000).unstack()

# NMF Solution for different number of latent factors
# Best Iteration: i=0, Public LB: 550
for i in range(1):
    A,Y=nmf(mat,i+2,max_iter=10)
    pred=pd.DataFrame(A.dot(Y),index=mat.index,columns=mat.columns).stack().to_frame('pred').reset_index()
    pred.index=pred['user_id'].astype(str)+'_'+pred['hero_id'].astype(str)
    store['test1']['pred%d'%i]=store['test1']['id'].map(pred['pred'])
    submission=store['test1'].copy()
    submission['kda_ratio']=submission['pred%d'%i]
    submission[['id','kda_ratio']].to_csv('02.Submissions/3.NMF%d.csv'%i,index=False)
    
# Make mean equal to value obtained by LB Probing. 
# Performance degrades in this step
submission['kda_ratio']=15+1.0064*submission['pred0']
submission[['id','kda_ratio']].to_csv('02.Submissions/3.NMF_Norm.csv',index=False) 

# Get user, iteam means solution
# Public LB Score: 555
a=pd.concat([store['train9'],store['test9']])
a['kda_ratio']=np.clip(a['kda_ratio'],2000,6000)
user_mean1=a.groupby(['user_id'])['kda_ratio'].mean()
hero_mean1=a.groupby(['hero_id'])['kda_ratio'].mean()
user_mean2=a['kda_ratio']-a['hero_id'].map(hero_mean1)
user_mean2=user_mean2.groupby(a['user_id']).mean()
hero_mean2=a['kda_ratio']-a['user_id'].map(user_mean1)
hero_mean2=hero_mean2.groupby(a['hero_id']).mean()

#Mean encoding on validation dataset.
b=store['train1'].copy()
b['F1']=b['user_id'].map(user_mean1)
b['F2']=b['user_id'].map(user_mean2)
b['F3']=b['hero_id'].map(hero_mean1)
b['F4']=b['hero_id'].map(hero_mean2)

#Run regression on validation to get the optimal parameters
import statsmodels.api as sm
Y = b['kda_ratio']
X = b[['F1','F2','F3','F4']]
model = sm.OLS(Y,sm.add_constant(X))
results = model.fit()
results.summary()

#Score test data using above parameters
c=store['test1'].copy()
c['F1']=c['user_id'].map(user_mean1)
c['F2']=c['user_id'].map(user_mean2)
c['F3']=c['hero_id'].map(hero_mean1)
c['F4']=c['hero_id'].map(hero_mean2)
c['kda_ratio']=c['F1']+c['F4']
c['kda_ratio']=949.4+0.2383*c['F1']+0.8295*c['F2']+0.4924*c['F3']+0.6333*c['F4']
c[['id','kda_ratio']].to_csv('02.Submissions/4.UserHeroMean4.csv',index=False)
c['kda_ratio']=(c['F1']+c['F2']+c['F3']+c['F4'])/2
c[['id','kda_ratio']].to_csv('02.Submissions/4.UserHeroMean6.csv',index=False)
c['kda_ratio']=c['F1']+c['F4']
c[['id','kda_ratio']].to_csv('02.Submissions/4.UserHeroMean7.csv',index=False)
c['kda_ratio']=c['F2']+c['F3']
c[['id','kda_ratio']].to_csv('02.Submissions/4.UserHeroMean8.csv',index=False)

#Make enseble of NMF0 and Mean encoding solution
#Public LB: 549
a=pd.read_csv('02.Submissions/3.NMF0.csv')
b=pd.read_csv('02.Submissions/4.UserHeroMean6.csv')
c=pd.merge(a,b,on=['id'])
c['kda_ratio']=0.5*c['kda_ratio_x']+0.5*c['kda_ratio_y']
c[['id','kda_ratio']].to_csv('02.Submissions/5.Ensemble3.csv',index=False)

#Interpolate Using ranks
# a=pd.concat([store['train9'],store['train1'],store['test9'],store['test1']]).reset_index()
# a['rank']=a.groupby(['user_id'])['num_games'].rank(ascending=False,method='first')
# a.set_index(['user_id','rank'],inplace=True)
# mat_train=a['kda_ratio'].unstack()
# mat_train.iloc[:,9]=mat_train.iloc[:,9].fillna(mat_train.min(axis=1))
# mat_train.iloc[:,0]=mat_train.iloc[:,0].fillna(mat_train.max(axis=1))
# mat_train=mat_train.T.interpolate().T
# a['pred']=mat_train.stack()
# a=a[a['source']=='test1']
# a['kda_ratio']=a['pred']
# a[['id','kda_ratio']].to_csv('02.Submissions/6.FavouriteHeroInterpolation.csv',index=False)

