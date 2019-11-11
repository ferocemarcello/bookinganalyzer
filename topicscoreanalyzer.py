import csv
import os

import helper


def dividebynation(originfile):
    keywords = helper.getKeywords(originfile)
    for emotion in ['Bad']:
        print("begin " + emotion)
        for keyword in list(keywords.keys())[6:]:
            print(keyword)
            nationcluster={}
            try:
                csv_file = open('resources/gensim/noadj/outputtopsdocs/' + keyword + '_' + emotion.lower()+'/' +keyword + '_' + emotion.lower()+ '.csv', mode='r',
                            encoding="utf8", newline='\n')
                reader = csv.reader(csv_file, delimiter='|', quotechar='"')
                for row in reader:
                    nat = row[1]
                    if nat not in nationcluster.keys():
                        nationcluster[nat] = []
                    nationcluster[nat].append(row)
                for nat in nationcluster.keys():
                    if not os.path.exists(
                            'resources/gensim/noadj/outputtopsdocs/' + keyword + '_' + emotion.lower() + '/bycountry/'):
                        os.makedirs(
                            'resources/gensim/noadj/outputtopsdocs/' + keyword + '_' + emotion.lower() + '/bycountry/')
                    csv_file = open(
                        'resources/gensim/noadj/outputtopsdocs/' + keyword + '_' + emotion.lower() + '/bycountry/' + nat + '.csv',
                        mode='w', encoding="utf8",
                        newline='\n')
                    writer = csv.writer(csv_file, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    for r in nationcluster[nat]:
                        writer.writerow(r)
                    csv_file.close()
            except:
                None