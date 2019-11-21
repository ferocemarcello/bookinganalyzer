import csv
import time
import os
import numpy

import helper
def cluster(csv_reader):
    firstrow = next(csv_reader)
    maxlentokens = firstrow.count('')
    firstrow = firstrow[maxlentokens:]
    country_cluster_hotel={}
    country_cluster_tourist = {}
    clusters=[country_cluster_hotel, country_cluster_tourist]
    for row in csv_reader:
        country_hot=row[5]
        country_tour = row[1]
        countries=[country_hot,country_tour]
        values=row[maxlentokens:]
        values=list(map(int, values))
        for i in range(2):
            if countries[i] not in clusters[i].keys():
                clusters[i][countries[i]]={}
                clusters[i][countries[i]]['sum']=[0]*len(values)
                clusters[i][countries[i]]['count_rev']=0
            clusters[i][countries[i]]['count_rev']+=1
            clusters[i][countries[i]]['sum']=numpy.add(clusters[i][countries[i]]['sum'], values)
    for cc in [country_cluster_hotel, country_cluster_tourist]:
        for country in cc.keys():
            cc[country]['rel_freq']=[x/cc[country]['count_rev'] for x in cc[country]['sum']]
    return firstrow,country_cluster_hotel,country_cluster_tourist
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
                    tokens,country_cluster_hotel,country_cluster_tourist=cluster(csv_reader)
                csv_file.close()
            except:
                goforward=False
            if goforward:
                if not os.path.exists('resources/bow/country_freq/byhotelcountry/'):
                    os.makedirs('resources/bow/country_freq/byhotelcountry/')
                if not os.path.exists('resources/bow/country_freq/bytouristcountry/'):
                    os.makedirs('resources/bow/country_freq/bytouristcountry/')
                with open('resources/bow/country_freq/byhotelcountry/' + keyword + '_' + emotion.lower() + '.csv', mode='w') as file:
                    writer = csv.writer(file, delimiter='|', quotechar='"',
                                        quoting=csv.QUOTE_MINIMAL)
                    writer.writerow([''] * 2 + tokens)
                    for country in country_cluster_hotel.keys():
                        writer.writerow([country]+[country_cluster_hotel[country]['count_rev']]+list(map("{:.15f}".format, country_cluster_hotel[country]['rel_freq'])))
                file.close()
                with open('resources/bow/country_freq/bytouristcountry/' + keyword + '_' + emotion.lower() + '.csv', mode='w') as file:
                    writer = csv.writer(file, delimiter='|', quotechar='"',
                                        quoting=csv.QUOTE_MINIMAL)
                    writer.writerow([''] * 2 + tokens)
                    for country in country_cluster_tourist.keys():
                        writer.writerow([country]+[country_cluster_tourist[country]['count_rev']]+list(map("{:.15f}".format, country_cluster_tourist[country]['rel_freq'])))
                file.close()
            print('------------------------------------------------------')
            print(str(time.time() - start_time) + ' seconds to compute ' + keyword + ' ' + emotion)