import csv
import os
import time

import numpy

import helper
from indexmanager import get_country_to_code


def cluster(csv_reader):
    firstrow = next(csv_reader)
    maxlentokens = firstrow.count('')
    firstrow = firstrow[maxlentokens:]
    cluster_tourist_hotel = {}
    country_hotel_code = get_country_to_code()
    for row in csv_reader:
        id=row[0]
        country_hot=row[5]
        if row[1] == '':
            country_tour='no_country'
        else:
            country_tour = country_hotel_code[row[1]]
        countries=(country_tour,country_hot)
        values=row[maxlentokens:]
        values=list(map(int, values))
        if countries not in cluster_tourist_hotel.keys():
            cluster_tourist_hotel[countries]={}
            cluster_tourist_hotel[countries]['sum']=[0]*len(values)#sum calculated on the sentences
            cluster_tourist_hotel[countries]['count_rev']=0
            cluster_tourist_hotel[countries]['unique_reviews'] = set()
        if id in cluster_tourist_hotel[countries]['unique_reviews']:
            for i in range(len(values)):
                if values[i] == 1 and cluster_tourist_hotel[countries]['sum'][i] == 0:
                    cluster_tourist_hotel[countries]['sum'][i] = 1
        else:
            cluster_tourist_hotel[countries]['unique_reviews'].add(id)
            cluster_tourist_hotel[countries]['sum'] = numpy.add(cluster_tourist_hotel[countries]['sum'], values)
    for countries in cluster_tourist_hotel.keys():
        #count calculated on the unique id
        cluster_tourist_hotel[countries]['count_rev']=len(list(cluster_tourist_hotel[countries]['unique_reviews']))
        cluster_tourist_hotel[countries]['rel_freq']=[x/cluster_tourist_hotel[countries]['count_rev'] for x in cluster_tourist_hotel[countries]['sum']]
    return firstrow, cluster_tourist_hotel

def do(originfile, all=False):
    if all:
        start_time = time.time()
        print('all ----- ' + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
        with open('resources/bow/all.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='|')
            tokens, cluster_tourist_hotel = cluster(csv_reader)
        csv_file.close()
        if not os.path.exists('resources/bow/tourist_hotel_country_freq/'):
            os.makedirs('resources/bow/tourist_hotel_country_freq/')
        with open('resources/bow/tourist_hotel_country_freq/all.csv',
                  mode='w') as file:
            writer = csv.writer(file, delimiter='|', quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)
            writer.writerow([''] * 2 + ['unique IDs'] + tokens)
            for country in cluster_tourist_hotel.keys():
                writer.writerow([country[0], country[1]] + [cluster_tourist_hotel[country]['count_rev']] + list(
                    map("{:.15f}".format, cluster_tourist_hotel[country]['rel_freq'])) + list(
                    cluster_tourist_hotel[country]['unique_reviews']))
        file.close()
        print('------------------------------------------------------')
        print(str(time.time() - start_time) + ' seconds to compute all')
    else:
        keywords = helper.getKeywords(originfile)
        for emotion in ['Good', 'Bad']:
            print("begin " + emotion)
            for keyword in list(keywords.keys()):
                start_time = time.time()
                goforward=True
                print(keyword + ' ---- ' + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
                try:
                    with open('resources/bow/'+keyword+'_'+emotion.lower()+'.csv') as csv_file:
                        csv_reader = csv.reader(csv_file, delimiter='|')
                        tokens,cluster_tourist_hotel=cluster(csv_reader)
                    csv_file.close()
                except:
                    goforward=False
                if goforward:
                    if not os.path.exists('resources/bow/tourist_hotel_country_freq/'):
                        os.makedirs('resources/bow/tourist_hotel_country_freq/')
                    with open('resources/bow/tourist_hotel_country_freq/' + keyword + '_' + emotion.lower() + '.csv', mode='w') as file:
                        writer = csv.writer(file, delimiter='|', quotechar='"',
                                            quoting=csv.QUOTE_MINIMAL)
                        writer.writerow([''] * 2 + ['unique IDs'] + tokens)
                        for country in cluster_tourist_hotel.keys():
                            writer.writerow([country[0],country[1]]+[cluster_tourist_hotel[country]['count_rev']]+list(map("{:.15f}".format, cluster_tourist_hotel[country]['rel_freq']))+list(cluster_tourist_hotel[country]['unique_reviews']))
                    file.close()
                print('------------------------------------------------------')
                print(str(time.time() - start_time) + ' seconds to compute ' + keyword + ' ' + emotion)