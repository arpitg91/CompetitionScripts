# Import required libraries and set folder path
import pandas as pd
import numpy as np
import os,sys,re
from sklearn.feature_extraction.text import TfidfVectorizer
os.chdir('C:/Users/arpit.goel/Documents/Projects/Kaggle/28_ResearchMatch/')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from itertools import product
ps = PorterStemmer()
stop_words=set(stopwords.words('english'))

validation_sets=[999]
# validation_sets=[5,8]
neg_sampling_ratio=1
th=0.275

def clean_text(x):
    x=str(x).lower()
    x=re.sub('[^a-z ]+', ' ',x)
    x=' '.join([ps.stem(y) for y in x.split(' ') if y not in stop_words])
    return x

#Import all rawdata
df_info_train=pd.read_csv('01.RawData/information_train.csv',delimiter='\t',parse_dates=['pub_date'],index_col=['pmid'])
df_info_test=pd.read_csv('01.RawData/information_test.csv',delimiter='\t',parse_dates=['pub_date'],index_col=['pmid'])
df_train=pd.read_csv('01.RawData/train.csv')
df_test=pd.read_csv('01.RawData/test.csv')
df_subm=pd.read_csv('01.RawData/sample_submission_eSUXEfp.csv')

#Parse the citations
df_train['ref_list_parse']=df_train['ref_list'].map(lambda x: map(int,re.sub('[^0-9,]','',x).split(',')))
df_train['num_citations']=df_train['ref_list_parse'].map(lambda x: len(x))

# Make flat file for citation indicator
def get_citation_master(df1,df2):
    citation_master=df1[['set','pub_date']].reset_index()
    citation_master=pd.merge(citation_master,citation_master,on=['set'],suffixes=['','_citation'])
    citation_master=citation_master[citation_master['pmid']!=citation_master['pmid_citation']]
    #This filter removed 529 citation records
    citation_master=citation_master[citation_master['pub_date']>=citation_master['pub_date_citation']]
    citation_master['sample']='test'
    # Create citation flag for train set
    if 'num_citations' in df2.columns:
        citations=[pd.Series(x[1]['ref_list_parse'],index=[x[1]['pmid']]*x[1]['num_citations']) for x in df_train.iterrows()]
        citations=pd.concat(citations).to_frame('pmid_citation')
        citations.index.name='pmid'
        citations=citations.reset_index()
        citations['flag_citation']=1
        citation_master=pd.merge(citation_master,citations,how='left',on=['pmid','pmid_citation'])
        citation_master['flag_citation']=citation_master['flag_citation'].fillna(0)

        #Keep 2 sets for validation
        #Keep only 10 percent of the negatives
        np.random.seed(1235)
        citation_master['random']=np.random.random(len(citation_master))
        citation_master['keep1']=citation_master['set'].isin(validation_sets)
        citation_master['keep2']=citation_master['random']<=neg_sampling_ratio
        citation_master=citation_master[citation_master[['keep1','keep2','flag_citation']].max(axis=1)>0]
        citation_master['sample']=np.where(citation_master['set'].isin(validation_sets),'valid','train')
    citation_master.drop(['keep1','keep2','random','pub_date','pub_date_citation'],axis=1,inplace=True,errors='ignore')
    return citation_master

master_train=get_citation_master(df_info_train,df_train)
master_test=get_citation_master(df_info_test,df_test)
master=pd.concat([master_train,master_test])
master_info=pd.concat([df_info_train,df_info_test])
master_info['snum']=np.arange(len(master_info))
master_info['abstract_article_title']=master_info['article_title']+' '+master_info['abstract']
#Make feature for common authors
authors=master_info['author_str'].map(lambda x: set(str(x).split(',')))
master['t_authors']=master['pmid'].map(authors)
master['t_authors_citation']=master['pmid_citation'].map(authors)
master['t_num_authors']=master['t_authors'].map(lambda x: len(x))
master['f_num_common_authors']=master[['t_authors','t_authors_citation']].apply(lambda x:len(x[0].intersection(x[1])),axis=1)
master.drop(['t_authors','t_authors_citation','t_num_authors'],axis=1,inplace=True)

#Make variable for datediff
master['f_date_diff']=(master['pmid'].map(master_info['pub_date'])-master['pmid_citation'].map(master_info['pub_date'])).dt.days

#Make variables based on text similarities
def get_text_similarities(col,ngram=1):
    master_info[col]=master_info[col].map(clean_text)
    vector1=master['pmid'].map(master_info[col])
    vector2=master['pmid_citation'].map(master_info[col])

    for n in range(ngram):
        tfidf_vectorizer=TfidfVectorizer(max_df=0.2,min_df=2,ngram_range=(n+1,n+1))
        tfidf_vectorizer.fit(master_info[col])
        print col,n,len(tfidf_vectorizer.vocabulary_.keys())
        matrix1=tfidf_vectorizer.transform(vector1)
        matrix2=tfidf_vectorizer.transform(vector2)
        matches=matrix1.multiply(matrix2)
        master['f_max_%s_%d'%(col,n+1)]=matches.max(axis=1).toarray()[:,0]
        master['f_sum_%s_%d'%(col,n+1)]=np.array(matches.sum(axis=1))
        master['f_cnt_%s_%d'%(col,n+1)]=np.array((matches>0).sum(axis=1))

get_text_similarities('article_title',2)
get_text_similarities('abstract')
get_text_similarities('abstract_article_title')

#Make train validation split
train=master[master['sample']=='train']
valid=master[master['sample']=='valid']
test=master[master['sample']=='test']

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
idv=['f_num_common_authors','f_date_diff','f_max_article_title','f_sum_article_title','f_cnt_article_title',\
     'f_max_abstract','f_sum_abstract','f_cnt_abstract']
     # 'f_max_abstract_article_title','f_sum_abstract_article_title','f_cnt_abstract_article_title'
idv=[x for x in train.columns if x[:2]=='f_']
clf=DecisionTreeClassifier(max_depth=5, min_samples_split=200, min_samples_leaf=100,random_state=1234, class_weight={0:1.0/neg_sampling_ratio,1:1})
clf.fit(train[idv],train['flag_citation'])

# Print model performance
# print roc_auc_score(train['flag_citation'],clf.transform(train[idv])[:,1])
# print roc_auc_score(valid['flag_citation'],clf.transform(valid[idv])[:,1])
# print pd.Series(clf.feature_importances_,index=idv).sort_values(ascending=False)

# for th1,th2 in product([0.3],[0.3]):
#     # Make selection of recommendation
#     test.loc[:,'score']=clf.transform(test.loc[:,idv].fillna(0))[:,1]
#     test.loc[:,'score_rank']=test.groupby(['pmid'])['score'].rank(ascending=False)
#     test.loc[:,'th']=np.where(test.loc[:,'pmid'].map(master_info['set']).isin([7,11,6,1]),th1,th2)
#     test.loc[:,'predict1']=test.loc[:,'score']>test.loc[:,'th']
#     test.loc[:,'predict2']=test.loc[:,'score_rank']==1
#     test.loc[:,'predict']=test.loc[:,['predict1','predict2']].max(axis=1)
#     test=test.sort_values(by=['score'],ascending=False)

#     # Make final submission
#     output=test[test.loc[:,'predict']]
#     print th, len(output), len(output['pmid'].drop_duplicates())
#     output['pmid_citation']="'"+output['pmid_citation'].astype(str)+"'"
#     output=output.groupby(['pmid'])['pmid_citation'].apply(lambda x: '['+','.join(x)+']')
#     df_subm['ref_list']=df_subm['pmid'].map(output).fillna('['+df_subm['pmid'].astype(str)+']')
#     df_subm[['pmid','ref_list']].to_csv('02.Submission/10.Cutoff_%.03f_%.03f.csv'%(th1,th2),index=False)

test.loc[:,'score']=clf.predict(test.loc[:,idv].fillna(0))
test.loc[:,'predict']=clf.predict_proba(test.loc[:,idv].fillna(0))[:,1]
test.loc[:,'rank']=test.groupby('pmid')['predict'].rank(ascending=False)
output=test[((test.loc[:,'score']==1)|(test.loc[:,'rank']==1))]
output['pmid_citation']="'"+output['pmid_citation'].astype(str)+"'"
output=output.groupby(['pmid'])['pmid_citation'].apply(lambda x: '['+','.join(x)+']')
df_subm['ref_list']=df_subm['pmid'].map(output).fillna('['+df_subm['pmid'].astype(str)+']')
df_subm[['pmid','ref_list']].to_csv('02.Submission/11.Predict.csv',index=False)
