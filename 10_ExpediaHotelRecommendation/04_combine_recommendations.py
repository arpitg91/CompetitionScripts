import pandas as pd

previous_bookings = pd.read_csv('/data/arpit.goel/14_Expedia_Hotel_Recommendation/03.ProcessedData/user_bookings.csv',header=None,names=['user_id','previous_bookings'])
previous_visits = pd.read_csv('/data/arpit.goel/14_Expedia_Hotel_Recommendation/03.ProcessedData/user_visits.csv',header=None,names=['user_id','previous_visits'])
test_data = pd.read_csv('/data/arpit.goel/14_Expedia_Hotel_Recommendation/01.RawData/02.Unzipped/test.csv',usecols=['id','user_id'])
scored_data = pd.read_csv('/data/arpit.goel/14_Expedia_Hotel_Recommendation/04.Modelling/01.GlobalBias/test_scored.csv')
scored_data = pd.merge(scored_data,test_data,on='id')
scored_data = pd.merge(scored_data,previous_bookings,on='user_id',how='left')
scored_data = pd.merge(scored_data,previous_visits,on='user_id',how='left')

def get_all_recommendations(series):
    series =  series.fillna('')
    previous_visits = series['previous_visits'].split(' ')
    previous_bookings = series['previous_bookings'].split(' ')
    hotel_cluster = series['hotel_cluster'].split(' ')
    final_cluster = previous_bookings[:]
    final_cluster += filter(lambda x: x not in previous_bookings,previous_visits)
    final_cluster += filter(lambda x: x not in previous_visits+previous_bookings,hotel_cluster)
    final_cluster = filter(lambda x: len(x)>0,final_cluster)
    return ' '.join(final_cluster[:min(len(final_cluster),5)])
    
scored_data['hotel_cluster'] = scored_data.apply(get_all_recommendations,axis=1)
scored_data[['id','hotel_cluster']].to_csv('/data/arpit.goel/14_Expedia_Hotel_Recommendation/04.Modelling/01.GlobalBias/test_scored_with_user_history.csv',index=False)

