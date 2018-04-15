import lightgbm as lgb

idv=store['train1_x'].columns

sample=store['train1_x'].index.to_series().drop_duplicates()
train_lgb=lgb.Dataset(store['train1_x'].loc[sample,idv].fillna(-99999),store['train1_y'].loc[sample,'tgt'])

def model_lgb(params):
    model=lgb.train(params, train_lgb)
    ins_pred=pd.Series(model.predict(store['train1_x'].loc[:,idv].fillna(-99999)))
    oos_pred=pd.Series(model.predict(store['val1_x'].loc[:,idv].fillna(-99999)))
    ins_error=np.sqrt(mean_squared_error(store['train1_y'].loc[:,'tgt'].values,ins_pred))
    oos_error=np.sqrt(mean_squared_error(store['val1_y'].loc[:,'tgt'].values,oos_pred))
    #print 'RMSLE: ',np.array([ins_error,oos_error])
    var_importance=pd.Series(model.feature_importance(),index=idv)
    #var_importance=var_importance[var_importance>0]
    print var_importance.sort_values(ascending=False).head()
    test_score=pd.Series(np.expm1(model.predict(store['test1_x'].loc[:,idv])),index=store['test1_x'].index.tolist()).to_frame('visitors')
    test_score['visit_date']=store['test1_x']['day'].map(lambda x: date(2017,4,23)+timedelta(days=x))
    return np.array([ins_error,oos_error]),var_importance,test_score
   
params = {'objective': 'regression','metric': 'rmse','boosting': 'gbdt','max_depth': 5,'num_leaves': 2**3,
          'learning_rate': 0.01,'num_rounds': 400,'min_data_in_leaf':500,
          'bagging_fraction': 0.8,'feature_fraction': 0.8}
output=model_lgb(params)

# for param in [100,200,300,500,1000]:
#     params = {'objective': 'regression','metric': 'rmse','boosting': 'gbdt','max_depth': 5,'num_leaves': 2**5,
#               'learning_rate': 0.05,'num_rounds': param,'min_data_in_leaf':500,
#               'bagging_fraction': 0.8,'feature_fraction': 0.8}
#     print param,model_lgb(params)[0]
    
    
    
 
 
 from sklearn.linear_model import LinearRegression
from itertools import product
np.set_printoptions(precision=3,suppress=True)

def model_train(params):
    ins_pred=[]
    oos_pred=[]
    ins_tgt=[]
    oos_tgt=[]

    day_errors=[]
    var_importance={}
    test_scores=[]
    for i in range(39):
        ins_msk=store['train_y'].iloc[:,i].fillna(0)>0
        oos_msk=store['val_y'].iloc[:,i].fillna(0)>0
            
        if params!=[]:
            idv=['%s_visitors_%d'%(x,y) for x,y in product(['min','max','avg','med','cnt'],[1,2,3,7,14,21,35,56,91,147])]
            idv+=[x for x in store['train_x'].columns if 'wd%d'%(6-i%7) in x]
            idv+=['air_reserve_%d'%i]
            train_lgb=lgb.Dataset(store['train_x'].loc[ins_msk,idv],store['train_y'].loc[ins_msk,:].iloc[:,i])
            model=lgb.train(params, train_lgb)
            ins_pred.append(pd.Series(model.predict(store['train_x'].loc[ins_msk,idv])))
            oos_pred.append(pd.Series(model.predict(store['val_x'].loc[oos_msk,idv])))
        else:
            idv=['med_visitors_56','avg_visitors_wd%d_13'%(6-i%7)]
            model=LinearRegression()
            ins_vars=store['train_x'].loc[ins_msk,idv]
            ins_vars['med_visitors_56']=ins_vars['med_visitors_56'].replace(-99999,2.86)
            ins_vars['avg_visitors_wd%d_13'%(6-i%7)]=np.where(ins_vars['avg_visitors_wd%d_13'%(6-i%7)]==-99999,ins_vars['med_visitors_56'],ins_vars['avg_visitors_wd%d_13'%(6-i%7)])
            oos_vars=store['val_x'].loc[oos_msk,idv]
            oos_vars['med_visitors_56']=oos_vars['med_visitors_56'].replace(-99999,2.86)
            oos_vars['avg_visitors_wd%d_13'%(6-i%7)]=np.where(oos_vars['avg_visitors_wd%d_13'%(6-i%7)]==-99999,oos_vars['med_visitors_56'],oos_vars['avg_visitors_wd%d_13'%(6-i%7)])
            model.fit(ins_vars,store['train_y'].loc[ins_msk,:].iloc[:,i])
            ins_pred.append(pd.Series(model.predict(ins_vars)))
            oos_pred.append(pd.Series(model.predict(oos_vars)))
            print 'COEF: ',np.array([model.intercept_]),model.coef_

        ins_tgt.append(pd.Series(store['train_y'].loc[ins_msk,:].iloc[:,i].values))
        oos_tgt.append(pd.Series(store['val_y'].loc[oos_msk,:].iloc[:,i].values))
        ins_error=np.sqrt(mean_squared_error(ins_tgt[-1],ins_pred[-1]))
        oos_error=np.sqrt(mean_squared_error(oos_tgt[-1],oos_pred[-1]))
        print 'RMSLE: ',np.array([ins_error,oos_error])
        day_errors.append([ins_error,oos_error])
        var_importance[i]=pd.Series(model.feature_importance(),index=idv)
        test_score=pd.Series(np.expm1(model.predict(store['test_x'].loc[:,idv])),index=store['test_x'].index.tolist()).to_frame('visitors')
        test_score['visit_date']=date(2017,4,23)+timedelta(days=i)
        test_scores.append(test_score)
    ins_error=np.sqrt(mean_squared_error(pd.concat(ins_tgt),pd.concat(ins_pred)))
    oos_error=np.sqrt(mean_squared_error(pd.concat(oos_tgt),pd.concat(oos_pred)))
    day_errors.append([ins_error,oos_error])
    day_errors=pd.DataFrame(day_errors,columns=['train','test'])
    var_importance=pd.DataFrame(var_importance).unstack()
    var_importance=var_importance[var_importance>0]
    test_scores=pd.concat(test_scores)
    return day_errors,var_importance,test_scores

#a=model_train([])

import lightgbm as lgb
params = {'objective': 'regression','metric': 'rmse','boosting': 'gbdt','max_depth': 10,'num_leaves': 2**5,
              'learning_rate': 0.05,'num_rounds': 100,
        'bagging_fraction': 0.8,'feature_fraction': 0.8}

output=model_train(params)