import csv
import os
import time

import helper


def read_table(filepath):
    cluster_tourist_hotel={}
    with open(filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='|')
        firstrow = next(csv_reader)
        lenprevtokens = firstrow.count('')
        firstrow = firstrow[lenprevtokens:]
        for row in csv_reader:
            countries = (row[0], row[1])
            count=int(row[2])
            row=list(map(float,row[lenprevtokens:]))
            cluster_tourist_hotel[countries]={}
            cluster_tourist_hotel[countries]['tokens']={}
            for tok in firstrow:
                i=firstrow.index(tok)
                cluster_tourist_hotel[countries]['count_rev'] = count
                cluster_tourist_hotel[countries]['tokens'][tok]=row[i]
    csv_file.close()
    return cluster_tourist_hotel


def get_diff_table(good_table, bad_table):
    diff_table={}
    pos=0
    neg=0
    for countries in bad_table.keys():
        if countries in good_table.keys():
            diff_table[countries]={}
            diff_table[countries]['tokens'] = {}
            diff_table[countries]['count_rev']=bad_table[countries]['count_rev']+good_table[countries]['count_rev']
            for tok in bad_table[countries]['tokens'].keys():
                if tok in good_table[countries]['tokens'].keys():
                    diff_table[countries]['tokens'][tok]={}
                    diff_table[countries]['tokens'][tok]['good'] = good_table[countries]['tokens'][tok]
                    diff_table[countries]['tokens'][tok]['bad'] = bad_table[countries]['tokens'][tok]
                    diff_table[countries]['tokens'][tok]['diff'] = good_table[countries]['tokens'][tok]-bad_table[countries]['tokens'][tok]
    return diff_table


def do(originfile):
    keywords = helper.getKeywords(originfile)
    for keyword in list(keywords.keys()):
        start_time = time.time()
        goforward = True
        print(keyword + ' ---- ' + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
        try:
            good_tab = read_table('resources/bow/tourist_hotel_country_freq/' + keyword + '_good.csv')
            bad_table = read_table('resources/bow/tourist_hotel_country_freq/' + keyword + '_bad.csv')
        except:
            goforward = False
        if goforward:
            diff_table=get_diff_table(good_tab,bad_table)
            if not os.path.exists('resources/bow/tourist_hotel_country_freq/diff/'):
                os.makedirs('resources/bow/tourist_hotel_country_freq/diff/')
            with open('resources/bow/tourist_hotel_country_freq/diff/'+ keyword+ '.csv',
                      mode='w') as file:
                writer = csv.writer(file, delimiter='|', quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['Tourist_Country_Index','Tourist_Country','Hotel_Country_Index','Hotel_Country','Sum_of_Count_of_Reviews','Token_Index','Token','Token_Frequence_in_Good','Token_Frequence_in_Bad','Difference'])
                for countries in diff_table.keys():
                    for tok in diff_table[countries]['tokens'].keys():
                        writer.writerow([0,countries[0],0,countries[1],diff_table[countries]['count_rev'],0,tok,"{:.15f}".format(diff_table[countries]['tokens'][tok]['good']),"{:.15f}".format(diff_table[countries]['tokens'][tok]['bad']),"{:.15f}".format(diff_table[countries]['tokens'][tok]['diff'])])
            file.close()
        print('------------------------------------------------------')
        print(str(time.time() - start_time) + ' seconds to compute ' + keyword)