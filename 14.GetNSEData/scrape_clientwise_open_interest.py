import requests
import pandas as pd
from datetime import date
import urllib
import os

def get_files(dates):
    files=os.listdir('01.DailyData/')
    files=[x.replace('.csv','') for x in files if 'csv' in x]
    for date in dates:
        date1=date.strftime('%d%m%Y') 
        date2=date.strftime('%Y%m%d') 
        if date2 in files:
            continue
        url = 'https://www.nseindia.com/content/nsccl/fao_participant_oi_%s.csv'%date1
        r = requests.get(url)
        with open('01.DailyData/%s.csv'%date2,'wb') as f:
            f.write(r.text)

def get_master_file(dates):
    output=[]
    for date in dates:
        date1=date.strftime('%d%m%Y') 
        date2=date.strftime('%Y%m%d') 
        df=pd.read_csv('01.DailyData/%s.csv'%date2,skiprows=1,index_col=0)
        if df.shape[0]==0:
            continue
        df['date']=date
        output.append(df)
    master=pd.concat(output)
    master.to_csv('master.csv')
        
    
dates=pd.date_range(start=date(2017,1,1),end=date.today())
get_files(dates)
get_master_file(dates)