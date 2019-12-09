import csv
import os
import time

import db
import helper

conn=db.db_connection()
conn.connect()
dbo=db.db_operator(conn)
good=[a[0] for a in dbo.execute('select Good from reviews limit 1000;')]
bad=[a[0] for a in dbo.execute('select Bad from reviews limit 1000;')]

keywords = helper.getKeywords('booking_keywords.txt')
diff_tables={}
validkeywords=[]
for keyword in list(keywords.keys()):
    start_time = time.time()
    goforward = True
    if keyword in ['breakfast', 'bedroom', 'bathroom' ,'location']:
        print(keyword + ' ---- ' + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
        try:
            '''query='CREATE TABLE masterthesis.'+keyword+'_diff' + '(Tourist_Country_Index VARCHAR(45) NOT NULL, Tourist_Country VARCHAR(45) NOT NULL, Hotel_Country_Index VARCHAR(45) NOT NULL, Hotel_Country VARCHAR(45) NOT NULL, Total_number_of_unique_reviews VARCHAR(45) NOT NULL, Token_Index VARCHAR(45) NOT NULL, Token VARCHAR(45) NOT NULL, Token_Frequence_in_Good VARCHAR(45) NOT NULL, Token_Frequence_in_Bad VARCHAR(45) NOT NULL, Difference VARCHAR(45) NOT NULL, PRIMARY KEY (Tourist_Country_Index, Hotel_Country_Index, Token_Index));'
            #'LOAD DATA LOCAL INFILE \'resources/bow/tourist_hotel_country_freq/diff/filtered/withcomb/reduced/location.csv\' INTO TABLE location_diff_reduced FIELDS TERMINATED BY \'|\' ENCLOSED BY \'"\' LINES TERMINATED BY \''+'n'+' IGNORE 1 ROWS;'
            '''
            query = 'CREATE TABLE masterthesis.' + keyword + '_diff_filtered_intersection_only ' + \
                    '(Country_of_origin VARCHAR(45) NOT NULL, Country_of_destination VARCHAR(45) NOT NULL, ' \
                    'Token_index SMALLINT NOT NULL, Frequence_difference VARCHAR(45),' \
                    ' PRIMARY KEY (Country_of_origin, Country_of_destination,Token_index));'
            dbo.execute(query)
            #with open('resources/bow/tourist_hotel_country_freq/diff/filtered' + keyword + '_'+emotion+'.csv') as csv_file:
            '''csv_reader = csv.reader(csv_file, delimiter='|')
            firstrow = next(csv_reader)
            csv_file.close()
            firstrow=firstrow[3:]'''
            '''query='CREATE TABLE masterthesis.'+keyword+'_diff_filtered_intersection_only '+ \
                  '(Country_of_origin VARCHAR(45) NOT NULL, Country_of_destination VARCHAR(45) NOT NULL, ' \
                  'Token_index SMALLINT NOT NULL, Frequence_difference VARCHAR(45),' \
                  ' PRIMARY KEY (Country_of_origin, Country_of_destination,Token_index));'
            '''
            '''for field in firstrow:
                query+=' `'+field+'` VARCHAR(15) NOT NULL, '
            '''
            #dbo.execute(query)
            '''query='LOAD DATA LOCAL INFILE \''+os.getcwd()+'/resources/bow/tourist_hotel_country_freq/'+keyword+\
                  '_'+emotion+'.csv\''+' INTO TABLE '+keyword+'_'+emotion+'_bow FIELDS TERMINATED BY \'|\' ENCLOSED BY \'"\' LINES TERMINATED BY \'\n\' IGNORE 1 ROWS;'
            dbo.execute(query)'''

        except Exception as e:
            goforward = False
        print('------------------------------------------------------')
        print(str(time.time() - start_time) + ' seconds to build the difference table for ' + keyword)
conn.disconnect()