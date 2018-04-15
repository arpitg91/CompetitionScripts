import pandas as pd
from dateutil.parser import parse
import csv, logging, sys
import numpy as np

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.info("Starting Process: python " + ' '.join(sys.argv))

# Define Columns for Global Bias
columns = map(str.upper,['hotel_continent','hotel_country','hotel_market','srch_destination_type_id'])

# Read the frequency files
threshold = 5000
bias_columns = [['HOTEL_CONTINENT','HOTEL_COUNTRY','HOTEL_MARKET','ADVANCE_BOOKING','CHECK_IN_MONTH'],['GROUP_SIZE','OCCUPANCY','STAY_TYPE','IS_MOBILE'],\
        ['USER_LOCATION_COUNTRY','USER_LOCATION_REGION','USER_LOCATION_CITY','CUSTOMER_TYPE']]
        
global_bias_files = '/data/arpit.goel/14_Expedia_Hotel_Recommendation/04.Modelling/01.GlobalBias/GlobalBias{0}.csv'

# Take Frequency at all levels
p_matrix = [[] for x in bias_columns]

for i,bias_column in enumerate(bias_columns):
    global_bias = pd.read_csv(global_bias_files.format(i),dtype=str)
    global_bias['COUNT'] = global_bias['COUNT'].astype(np.int64)
    for j in range(len(bias_column)):
        p = global_bias.groupby(bias_column[:j+1]+['HOTEL_CLUSTER'])['COUNT'].sum().unstack().fillna(0)
        p = p[p.sum(axis=1)>threshold]
        p_matrix[i].append(dict((p.T/p.sum(axis=1))))
        
        
default_vector =  global_bias.groupby('HOTEL_CLUSTER')['COUNT'].sum()
default_vector =   default_vector / default_vector.sum()
# Read and score test data
test_data = '/data/arpit.goel/14_Expedia_Hotel_Recommendation/01.RawData/02.Unzipped/test.csv'
scored_data = '/data/arpit.goel/14_Expedia_Hotel_Recommendation/04.Modelling/01.GlobalBias/test_scored.csv'

infile1 = open(test_data, 'rb')
outfile = open(scored_data,'wb')

reader = csv.reader(infile1, delimiter=',')
writer = csv.writer(outfile, delimiter=',')
writer.writerow(['id','hotel_cluster'])
header = reader.next()+['GROUP_SIZE','OCCUPANCY','STAY_TYPE','ADVANCE_BOOKING','CUSTOMER_TYPE','CHECK_IN_MONTH']
header_dict = dict(zip(map(str.upper,header),range(len(header))))

for row in reader:
    if reader.line_num%10000==0:
        logging.info('%d Rows Scored'%reader.line_num)
        
    CUSTOMER_TYPE = 'INTERNATIONAL' if row[header_dict['user_location_country'.upper()]]==row[header_dict['hotel_country'.upper()]] else 'DOMESTIC'
    num_adults = max(int(row[header_dict['srch_adults_cnt'.upper()]]),1)
    num_children = max(int(row[header_dict['srch_children_cnt'.upper()]]),0)
    total_people = 1.0*(num_adults+num_children)
    num_rooms = min(max(int(row[header_dict['srch_rm_cnt'.upper()]]),1),total_people)
    
    GROUP_SIZE = 'GROUP' if num_adults>2 else 'INDIVIDUAL/COUPLE+KID' if num_children > 0 else 'COUPLE' if num_adults == 2 else 'INDIVIDUAL'
    OCCUPANCY = '1' if num_rooms == total_people else '3' if num_rooms < 0.5*total_people else '2'

    try:
        booking_date = parse(row[header_dict['date_time'.upper()]])
        checkin_date = parse(row[header_dict['srch_ci'.upper()]])
        checkout_date = parse(row[header_dict['srch_co'.upper()]])
        STAY_TYPE = "EXTENDED" if (checkout_date-checkin_date).days > 3 else "BRIEF"
        ADVANCE_BOOKING = '<15DAYS' if (checkin_date-booking_date).days < 15 else '15-60DAYS' if (checkin_date-booking_date).days < 60 else '60+DAYS'
        CHECK_IN_MONTH = str(checkin_date.month)
    except:
        print row[header_dict['date_time'.upper()]], row[header_dict['srch_ci'.upper()]], row[header_dict['srch_co'.upper()]]
        STAY_TYPE = "BRIEF"
        ADVANCE_BOOKING = '15-60DAYS'
        CHECK_IN_MONTH = 12
        
    row += [GROUP_SIZE,OCCUPANCY,STAY_TYPE,ADVANCE_BOOKING,CUSTOMER_TYPE,CHECK_IN_MONTH]
        
    predictions_vectors = [default_vector for x in bias_columns]
        
    for i,bias_column in enumerate(bias_columns):
        test_vector = [row[header_dict[local_bias_column]] for local_bias_column in bias_column]
        search_space = len(bias_column)
        while search_space >=0:
            if tuple(test_vector[0:search_space]) in p_matrix[i][search_space-1].keys():
                predictions_vector = p_matrix[i][search_space-1][tuple(test_vector[0:search_space])]
                predictions_vector = predictions_vector.sort_index()
                predictions_vectors[i] = predictions_vector
                search_space = -1
            else:
                search_space = search_space-1

    predictions_vector = pd.concat(predictions_vectors,axis=1).product(axis=1)
    predictions_vector = predictions_vector/predictions_vector.sum()
    predictions_vector = ' '.join(list(predictions_vector.sort_values(ascending=False).head().index))
    writer.writerow([row[header_dict['ID']],predictions_vector])
    
infile1.close()
outfile.close()
