import requests, zipfile, StringIO,urllib,os,glob
import pandas as pd
from datetime import date

def get_bhav(dates,seg):
    loc='1.FO/' if seg=='fo' else '2.EQ/'
    prefix='fo' if seg=='fo' else 'cm'
    files=os.listdir(loc)
    files=[x.replace('bhav.csv','').replace(prefix,'') for x in files if 'csv' in x]
    for date in dates:
        date1=date.strftime('%Y/%b').upper()
        date2=date.strftime('%d%b%Y').upper()
        if date2 in files:
            continue
        if seg=='fo':
            url='https://www.nseindia.com/content/historical/DERIVATIVES/%s/fo%sbhav.csv.zip'%(date1,date2)
        else:
            url='https://www.nseindia.com/content/historical/EQUITIES/%s/cm%sbhav.csv.zip'%(date1,date2)
        try:
            r = requests.get(url, stream=True)
            z = zipfile.ZipFile(StringIO.StringIO(r.content))
            z.extractall(loc)
        except:
            f=open('%s/%s%sbhav.csv'%(loc,prefix,date2),'wb')
            f.close()
            
dates=pd.date_range(start=date(2017,1,1),end=date.today())
get_bhav(dates,'eq')
get_bhav(dates,'fo')

#Get Master for Equity Files
df_eq=pd.concat([pd.read_csv(x) for x in glob.glob('2.EQ/*') if len(open(x,'rb').readlines())>1])
df_eq['TIMESTAMP']=pd.to_datetime(df_eq['TIMESTAMP'].str.title(),format='%d-%b-%Y')
df_eq=df_eq.sort_values(by=['SYMBOL','TIMESTAMP'])
df_eq=df_eq[df_eq['SERIES']=='EQ']
df_eq.set_index(['SYMBOL','TIMESTAMP'],inplace=True)
df_eq['CLOSE'].unstack().to_csv('04.EQMaster.csv')
