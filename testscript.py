import csv
import os
import time

import db
import helper

conn=db.db_connection()
conn.connect()
dbo=db.db_operator(conn)
keywords = helper.getKeywords('booking_keywords.txt')
diff_tables={}
validkeywords=[]
cd=os.getcwd()
'''
            query = 'CREATE TABLE masterthesis.' + keyword + '_diff_filtered_intersection_only ' + \
                    '(Country_of_origin VARCHAR(45) NOT NULL, Country_of_destination VARCHAR(45) NOT NULL, ' \
                    'Token_index SMALLINT NOT NULL, Frequence_difference VARCHAR(45),' \
                    ' PRIMARY KEY (Country_of_origin, Country_of_destination,Token_index));'
            dbo.execute(query)
            #with open('resources/bow/tourist_hotel_country_freq/diff/filtered' + keyword + '_'+emotion+'.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='|')
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
for keyword in list(keywords.keys())+['all']:
    if keyword!='all':
        '''if os.path.isfile('resources/bow/tourist_hotel_country_freq/diff/'+keyword+'.csv'):
            start_time = time.time()
            print(keyword + ' ---- ' + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
            #dbo.execute("DROP TABLE IF EXISTS  masterthesis."+keyword+'_diff')
            query='CREATE TABLE masterthesis.'+keyword+'_diff' + '(Tourist_Country_Index SMALLINT NOT NULL, Tourist_Country VARCHAR(45) NOT NULL, Hotel_Country_Index SMALLINT NOT NULL, Hotel_Country VARCHAR(45) NOT NULL, Total_number_of_unique_reviews MEDIUMINT NOT NULL, Token_Index SMALLINT NOT NULL, Token VARCHAR(45) NOT NULL, Token_Frequence_in_Good VARCHAR(17) NOT NULL, Token_Frequence_in_Bad VARCHAR(17) NOT NULL, Difference DOUBLE(16,15) NOT NULL, PRIMARY KEY (Tourist_Country_Index, Hotel_Country_Index, Token_Index));'
            dbo.execute(query)
            query="LOAD DATA LOCAL INFILE '"+cd+"/resources/bow/tourist_hotel_country_freq/diff/"+keyword+".csv' INTO TABLE masterthesis."+keyword+"_diff FIELDS TERMINATED BY '|' ENCLOSED BY '"+'"'+"' LINES TERMINATED BY '\\" +"r\\n' IGNORE 1 ROWS;"
            dbo.execute(query)
            print('------------------------------------------------------')
            print(str(time.time() - start_time) + ' seconds to build the difference table for ' + keyword)
            '''
        if os.path.isfile('resources/bow/tourist_hotel_country_freq/diff/filtered/all_separetely/' + keyword + '.csv'):
            start_time = time.time()
            print(keyword + ' ---- ' + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
            # dbo.execute("DROP TABLE IF EXISTS  masterthesis."+keyword+'_diff')
            query = 'CREATE TABLE masterthesis.' + keyword + '_diff_sep' + '(Country_Origin_Index SMALLINT NOT NULL, Country_Destination_Index SMALLINT NOT NULL, Number_unique_reviews MEDIUMINT NOT NULL, Token_Index SMALLINT NOT NULL, Frequence_Difference DOUBLE(16,15) NOT NULL, PRIMARY KEY (Country_Origin_Index, Country_Destination_Index, Token_Index));'
            #dbo.execute(query)
            query="CREATE TABLE `masterthesis`.`"+keyword+"_diff_sep_country_index` (`index` INT NOT NULL, `country` VARCHAR(45) NOT NULL, PRIMARY KEY (`index`));"
            dbo.execute(query)
            query="CREATE TABLE `masterthesis`.`"+keyword+"_diff_sep_token_index` (`index` INT NOT NULL, `token` VARCHAR(45) NOT NULL, PRIMARY KEY (`index`));"
            dbo.execute(query)
            query = "LOAD DATA LOCAL INFILE '" + cd + "/resources/bow/tourist_hotel_country_freq/diff/filtered/all_separetely/" + keyword + ".csv' INTO TABLE masterthesis." + keyword + "_diff_sep FIELDS TERMINATED BY '|' ENCLOSED BY '" + '"' + "' LINES TERMINATED BY '\\r\\n' IGNORE 1 ROWS;"
            #dbo.execute(query)
            print('------------------------------------------------------')
            print(str(time.time() - start_time) + ' seconds to build the difference table for ' + keyword)
conn.disconnect()