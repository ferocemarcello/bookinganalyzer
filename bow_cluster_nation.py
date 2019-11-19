import csv
import time
import os
import numpy

import helper
def cluster(csv_reader):
    firstrow = next(csv_reader)
    maxlentokens = firstrow.count('')
    firstrow = firstrow[maxlentokens:]
    topk = len(firstrow)
    country_cluster={}
    i=0
    for row in csv_reader:
        i+=1
        country=row[5]
        values=row[maxlentokens:]
        values=list(map(int, values))
        if country not in country_cluster.keys():
            country_cluster[country]={}
            country_cluster[country]['sum']=[0]*len(values)
            country_cluster[country]['count_rev']=0
        country_cluster[country]['count_rev']+=1
        country_cluster[country]['sum']=numpy.add(country_cluster[country]['sum'],values)
    for country in country_cluster.keys():
        country_cluster[country]['rel_freq']=[x/country_cluster[country]['count_rev'] for x in country_cluster[country]['sum']]
    return firstrow,country_cluster
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
                    tokens,country_cluster=cluster(csv_reader)
                csv_file.close()
            except:
                goforward=False
            if goforward:
                if not os.path.exists('resources/bow/country_freq/'):
                    os.makedirs('resources/bow/country_freq/')
                with open('resources/bow/country_freq/' + keyword + '_' + emotion.lower() + '.csv', mode='w') as file:
                    writer = csv.writer(file, delimiter='|', quotechar='"',
                                        quoting=csv.QUOTE_MINIMAL)
                    writer.writerow([''] * 1 + tokens)
                    for country in country_cluster.keys():
                        writer.writerow([country]+country_cluster[country]['rel_freq'])
                file.close()
            print('------------------------------------------------------')
            print(str(time.time() - start_time) + ' seconds to compute ' + keyword + ' ' + emotion)