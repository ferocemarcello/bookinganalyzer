import csv
import os
import time

import numpy

import helper


def cluster(csv_reader):
    firstrow = next(csv_reader)
    maxlentokens = firstrow.count('')
    firstrow = firstrow[maxlentokens:]
    cluster_tourist_hotel = {}
    for row in csv_reader:
        country_hot=row[5]
        country_tour = row[1]
        countries=(country_tour,country_hot)
        values=row[maxlentokens:]
        values=list(map(int, values))
        if countries not in cluster_tourist_hotel.keys():
            cluster_tourist_hotel[countries]={}
            cluster_tourist_hotel[countries]['sum']=[0]*len(values)
            cluster_tourist_hotel[countries]['count_rev']=0
        cluster_tourist_hotel[countries]['count_rev']+=1
        cluster_tourist_hotel[countries]['sum']=numpy.add(cluster_tourist_hotel[countries]['sum'], values)
    for countries in cluster_tourist_hotel.keys():
        cluster_tourist_hotel[countries]['rel_freq']=[x/cluster_tourist_hotel[countries]['count_rev'] for x in cluster_tourist_hotel[countries]['sum']]
    return firstrow, cluster_tourist_hotel

def do(originfile):
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
                    writer.writerow([''] * 3 + tokens)
                    for country in cluster_tourist_hotel.keys():
                        writer.writerow([country[0],country[1]]+[cluster_tourist_hotel[country]['count_rev']]+list(map("{:.15f}".format, cluster_tourist_hotel[country]['rel_freq'])))
                file.close()
            print('------------------------------------------------------')
            print(str(time.time() - start_time) + ' seconds to compute ' + keyword + ' ' + emotion)