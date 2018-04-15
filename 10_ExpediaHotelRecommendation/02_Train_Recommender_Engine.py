import csv, logging, sys
from dateutil.parser import parse
import pandas as pd

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.info("Starting Process: python " + ' '.join(sys.argv))

# Step 1. Keep only the successful bookings of the training data'
def make_bookings_data():
    training_data = '/data/arpit.goel/14_Expedia_Hotel_Recommendation/01.RawData/02.Unzipped/train.csv'
    # training_data = '/data/arpit.goel/14_Expedia_Hotel_Recommendation/01.RawData/03.Sample/head_10000_train.csv'
    bookings_data = '/data/arpit.goel/14_Expedia_Hotel_Recommendation/03.ProcessedData/train_bookings.csv'
    
    infile1 = open(training_data, 'rb')
    outfile = open(bookings_data, 'wb')
    
    reader = csv.reader(infile1, delimiter=',')
    writer = csv.writer(outfile, delimiter=',')
    
    header = map(str.upper,reader.next())
    writer.writerow(header)
    
    for row in reader:
        if row[18]=="1":
            writer.writerow(row)
    
    infile1.close()        
    outfile.close()        

# Step 2. Get Destination Bias from Training Data    
def get_destination_bias():
    bookings_data = '/data/arpit.goel/14_Expedia_Hotel_Recommendation/03.ProcessedData/train_bookings.csv'
    global_bias_files = '/data/arpit.goel/14_Expedia_Hotel_Recommendation/04.Modelling/01.GlobalBias/GlobalBias{0}.csv'

    bias_columns = [['HOTEL_CONTINENT','HOTEL_COUNTRY','HOTEL_MARKET','ADVANCE_BOOKING','CHECK_IN_MONTH'],['GROUP_SIZE','OCCUPANCY','STAY_TYPE','IS_MOBILE'],\
        ['USER_LOCATION_COUNTRY','USER_LOCATION_REGION','USER_LOCATION_CITY','CUSTOMER_TYPE']]
    
    infile1 = open(bookings_data, 'rb')
    outfiles = [open(global_bias_files.format(i), 'wb') for i in range(len(bias_columns))]
    reader = csv.reader(infile1, delimiter=',')
    writers = [csv.writer(file, delimiter=',') for file in outfiles]
    counters = [{} for i in range(len(bias_columns))]
    
    header = reader.next()+['GROUP_SIZE','OCCUPANCY','STAY_TYPE','ADVANCE_BOOKING','CUSTOMER_TYPE','CHECK_IN_MONTH']
    header_dict = dict(zip(header,range(len(header))))
    
    [writers[i].writerow(x+['HOTEL_CLUSTER','COUNT']) for i,x in enumerate(bias_columns)]
    
    for row in reader:
        # if reader.line_num == 5:
            # break
        if reader.line_num%50000==0:
            logging.info('%d Rows Read'%reader.line_num)
            
        CUSTOMER_TYPE = 'INTERNATIONAL' if row[header_dict['user_location_country'.upper()]]==row[header_dict['hotel_country'.upper()]] else 'DOMESTIC'
        num_adults = max(int(row[header_dict['srch_adults_cnt'.upper()]]),1)
        num_children = max(int(row[header_dict['srch_children_cnt'.upper()]]),0)
        total_people = 1.0*(num_adults+num_children)
        num_rooms = min(max(int(row[header_dict['srch_rm_cnt'.upper()]]),1),total_people)
        
        GROUP_SIZE = 'GROUP' if num_adults>2 else 'INDIVIDUAL/COUPLE+KID' if num_children > 0 else 'COUPLE' if num_adults == 2 else 'INDIVIDUAL'
        OCCUPANCY = '1' if num_rooms == total_people else '3' if num_rooms < 0.5*total_people else '2'
        
        booking_date = parse(row[header_dict['date_time'.upper()]])
        checkin_date = parse(row[header_dict['srch_ci'.upper()]])
        checkout_date = parse(row[header_dict['srch_co'.upper()]])
        
        STAY_TYPE = "EXTENDED" if (checkout_date-checkin_date).days > 3 else "BRIEF"
        ADVANCE_BOOKING = '<15DAYS' if (checkin_date-booking_date).days < 15 else '15-60DAYS' if (checkin_date-booking_date).days < 60 else '60+DAYS'
        CHECK_IN_MONTH = str(checkin_date.month)
        
        row += [GROUP_SIZE,OCCUPANCY,STAY_TYPE,ADVANCE_BOOKING,CUSTOMER_TYPE,CHECK_IN_MONTH]
        
        for i,bias_column in enumerate(bias_columns):
            data_point = ','.join([row[header_dict[local_bias_column]] for local_bias_column in bias_column+['HOTEL_CLUSTER']])
            counters[i][data_point] = counters[i].get(data_point,0)+1
        
    [writers[i].writerow(key.split(',')+[str(value)]) for i,counter in enumerate(counters) for key, value in counter.items()]
    infile1.close()        
    [outfile.close() for outfile in outfiles]


def get_booked_hotels(group):
    hotels = pd.Series(group['COUNT'].values,index=group['HOTEL_CLUSTER'].values)
    hotels = ' '.join(map(str,list(hotels.sort_values(ascending=False).head().index)))
    return hotels
    
def get_user_hotel_preference():
    training_data = pd.read_csv('/data/arpit.goel/14_Expedia_Hotel_Recommendation/01.RawData/02.Unzipped/train.csv',usecols=['user_id','hotel_cluster'])
    training_data.columns = map(str.upper,list(training_data.columns))
    bookings_data = pd.read_csv('/data/arpit.goel/14_Expedia_Hotel_Recommendation/03.ProcessedData/train_bookings.csv',usecols=map(str.upper,['user_id','hotel_cluster']))
    
    bookings_data['COUNT'] = 1
    previous_bookings = bookings_data.groupby(['USER_ID','HOTEL_CLUSTER'])[['COUNT']].sum().reset_index()
    previous_bookings = previous_bookings.groupby('USER_ID').apply(get_booked_hotels)

    training_data['COUNT'] = 1
    previous_visits = training_data.groupby(['USER_ID','HOTEL_CLUSTER'])[['COUNT']].sum().reset_index()
    previous_visits = previous_visits.groupby('USER_ID').apply(get_booked_hotels)
    
    previous_bookings.to_csv('/data/arpit.goel/14_Expedia_Hotel_Recommendation/03.ProcessedData/user_bookings.csv')
    previous_visits.to_csv('/data/arpit.goel/14_Expedia_Hotel_Recommendation/03.ProcessedData/user_visits.csv')
    
# make_bookings_data()
# get_destination_bias()
get_user_hotel_preference()