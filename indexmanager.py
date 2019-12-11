import csv
import operator
import traceback

import pycountry

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
    special_countries=list()
    country_to_code=dict()
    #https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes
    country_to_code['Antigua &amp; Barbuda']='ag'
    country_to_code['Bonaire St Eustatius and Saba']='bq'
    country_to_code['Cape Verde'] = 'cv'
    country_to_code['Central Africa Republic'] = 'cf'
    country_to_code['Cocos (K) I.'] = 'cc'
    country_to_code['CuraÃ§ao'] = 'cw'
    country_to_code['Democratic Republic of the Congo'] = 'cd'
    country_to_code['East Timor'] = 'tl'
    country_to_code['Equitorial Guinea'] = 'gq'
    country_to_code['France, Metro.'] = 'fr'
    country_to_code['Heard and McDonald Islands'] = 'hm'
    country_to_code['Laos'] = 'la'
    country_to_code['Netherlands Antilles'] = 'nt'
    country_to_code['North Korea'] = 'kp'
    country_to_code['Palestinian Territory'] = 'ps'
    country_to_code['Saint Vincent &amp; Grenadines'] = 'vc'
    country_to_code['SÃ£o TomÃ© and PrÃ­ncipe'] = 'st'
    country_to_code['Serbia and Montenegro'] = 'em'
    country_to_code['South Korea'] = 'kr'
    country_to_code['St. Helena'] = 'sh'
    country_to_code['St. Pierre and Miquelon'] = 'pm'
    country_to_code['Svalbard &amp; Jan Mayen'] = 'sj'
    country_to_code['Swaziland'] = 'sz'
    country_to_code['Turks &amp; Caicos Islands'] = 'tc'
    country_to_code['U.K. Virgin Islands'] = 'vg'
    country_to_code['U.S. Virgin Islands'] = 'vi'
    country_to_code['U.S.A.'] = 'us'

    for key in country_to_code.keys():
        special_countries.append(key)
    code_to_country=dict()
    for k,v in country_to_code.items():
        code_to_country[v]=k
    for cont in tourcountries:
        try:
            code=pycountry.countries.search_fuzzy(cont)[0].alpha_2.lower()
        except Exception as e:
            None
        if code not in code_to_country:
            code_to_country[code]=cont
        if cont not in country_to_code:
            country_to_code[cont]=code
    for cont in hotcountries:
        cname = (pycountry.countries.get(alpha_2=cont.upper())).name
        if cont not in code_to_country.keys():
            code_to_country[cont]=cname
            country_to_code[cname]=cont
    print("writing the indices")
    with open('resources/tourist_country_index.csv', mode='w') as file:
        writer = csv.writer(file, delimiter='|', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        for i in range(1,len(list(country_to_code.keys()))+1):
            writer.writerow([i,list(country_to_code.keys())[i-1]])
        i+=1
        writer.writerow([i,'no_country'])
        i += 1
        writer.writerow([i,''])
    file.close()
    with open('resources/hotel_country_index.csv', mode='w') as file:
        writer = csv.writer(file, delimiter='|', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        for i in range(1,len(list(code_to_country.keys()))+1):
            writer.writerow([i,list(code_to_country.keys())[i-1]])
        writer.writerow([i+1,'no_country'])
    file.close()
    with open('resources/country_to_code.csv', mode='w') as file:
        writer = csv.writer(file, delimiter='|', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        for key in [x[0] for x in sorted(country_to_code.items(), key=operator.itemgetter(1), reverse=False)]:
            writer.writerow([key, country_to_code[key]])
    file.close()
    with open('resources/code_to_country.csv', mode='w') as file:
        writer = csv.writer(file, delimiter='|', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        for key in [x[0] for x in sorted(code_to_country.items(), key=operator.itemgetter(0), reverse=False)]:
            writer.writerow([key, code_to_country[key]])
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
            writer.writerow([i, toklist[i-1]])
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

def get_country_to_code():
    country_hotel_code= {}
    with open('resources/country_to_code.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='|')
        for row in csv_reader:
            country_hotel_code[row[0]] = row[1]
    csv_file.close()
    return country_hotel_code


def update_token_index(tokenset):
    oldtokenset=set(get_token_index()['token_to_index'].keys())
    newtokenset=tokenset.union(oldtokenset)
    build_token_index(newtokenset)