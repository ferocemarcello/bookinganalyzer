import csv

from db import db_operator, db_connection


def build_country_indices():
    db = db_connection()
    db.connect()
    queryexecutor = db_operator(db)
    query = 'select distinct(Country) from masterthesis.reviews;'
    print("retrieving countries of tourists")
    tourcountries= [x[0] for x in queryexecutor.execute(query=query)][1:]

    query='select distinct(CountryID) from masterthesis.hotels;'
    print("retrieving countries of hotels")
    hotcountries = [x[0] for x in queryexecutor.execute(query=query)]
    db.disconnect()
    print("writing the indices")
    with open('resources/tourist_country_index.csv', mode='w') as file:
        writer = csv.writer(file, delimiter='|', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        for i in range(1,len(tourcountries+1)):
            writer.writerow([i,tourcountries[i]])
        writer.writerow([i+1,''])
    file.close()
    with open('resources/hotel_country_index.csv', mode='w') as file:
        writer = csv.writer(file, delimiter='|', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        for i in range(1,len(hotcountries+1)):
            writer.writerow([i,hotcountries[i]])
        writer.writerow([i+1,'no_location_of_hotel'])
    file.close()
    print("writing countries indices over")


def get_tourist_country_index():
    country_tourist_ind={}
    country_tourist_ind['index_to_country']={}
    country_tourist_ind['country_to_index']={}
    with open('resources/tourist_country_index.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='|')
        for row in csv_reader:
            country_tourist_ind['index_to_country'][int(row[0])]=row[1]
            country_tourist_ind['country_to_index'][row[1]] = int(row[0])
    csv_file.close()
    return country_tourist_ind

def get_hotel_country_index():
    country_hotel_ind={}
    country_hotel_ind['index_to_country']={}
    country_hotel_ind['country_to_index']={}
    with open('resources/hotel_country_index.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='|')
        for row in csv_reader:
            country_hotel_ind['index_to_country'][int(row[0])]=row[1]
            country_hotel_ind['country_to_index'][row[1]] = int(row[0])
    csv_file.close()
    return country_hotel_ind

def build_token_index(tokenset):
    toklist=list(tokenset)
    with open('resources/bow/tourist_hotel_country_freq/diff/token_index.csv', mode='w') as file:
        writer = csv.writer(file, delimiter='|', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        for i in range(1,len(toklist)+1):
            writer.writerow([i, toklist[i]])
    file.close()
def get_token_index():
    token_index={}
    token_index['token_to_index']={}
    token_index['index_to_token']={}
    with open('resources/bow/tourist_hotel_country_freq/diff/token_index.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='|')
        for row in csv_reader:
            token_index['index_to_token'][int(row[0])]=row[1]
            token_index['token_to_index'][row[1]] = int(row[0])
    return token_index