import csv
import os
import time

import helper
import indexmanager


def read_table(filepath):
    cluster_tourist_hotel={}
    with open(filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='|')
        firstrow = next(csv_reader)
        lenprevtokens = firstrow.count('')+1
        firstcellunid=len(firstrow)-lenprevtokens
        firstrow = firstrow[lenprevtokens:]
        for row in csv_reader:
            countries = (row[0], row[1])
            count=int(row[2])
            row=list(map(float,row[lenprevtokens:]))
            cluster_tourist_hotel[countries]={}
            cluster_tourist_hotel[countries]['tokens']={}
            cluster_tourist_hotel[countries]['unique_reviews']=set(list(map(str,list(map(int,row[firstcellunid:])))))
            cluster_tourist_hotel[countries]['count_rev'] = count
            for tok in firstrow:
                i=firstrow.index(tok)
                cluster_tourist_hotel[countries]['tokens'][tok]=row[i]
    csv_file.close()
    return cluster_tourist_hotel


def get_diff_table(good_table, bad_table, tokenset):
    diff_table={}
    for countries in bad_table.keys():
        if countries in good_table.keys():
            diff_table[countries]={}
            diff_table[countries]['tokens'] = {}
            diff_table[countries]['unique_reviews']=set()
            diff_table[countries]['unique_reviews']=bad_table[countries]['unique_reviews'].union(good_table[countries]['unique_reviews'])
            diff_table[countries]['count_rev']=len(list(diff_table[countries]['unique_reviews']))
            for tok in bad_table[countries]['tokens'].keys():
                if tok in good_table[countries]['tokens'].keys():
                    tokenset.add(tok)
                    diff_table[countries]['tokens'][tok]={}
                    diff_table[countries]['tokens'][tok]['good'] = good_table[countries]['tokens'][tok]
                    diff_table[countries]['tokens'][tok]['bad'] = bad_table[countries]['tokens'][tok]
                    diff_table[countries]['tokens'][tok]['diff'] = good_table[countries]['tokens'][tok]-bad_table[countries]['tokens'][tok]
    return diff_table


def do(originfile):
    keywords = helper.getKeywords(originfile)
    tokenset = set()
    diff_tables={}
    validkeywords=[]
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
            validkeywords.append(keyword)
            diff_table=get_diff_table(good_tab,bad_table,tokenset)
            diff_tables[keyword]=diff_table
        print('------------------------------------------------------')
        print(str(time.time() - start_time) + ' seconds to build the difference table for ' + keyword)
    print("start writing difference matrices")
    indexmanager.build_token_index(tokenset)
    token_index=indexmanager.get_token_index()
    for keyword in validkeywords:
        start_time = time.time()
        print(keyword + ' ---- ' + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
        country_tourist_ind = indexmanager.get_tourist_country_index()
        country_hotel_ind = indexmanager.get_hotel_country_index()
        if not os.path.exists('resources/bow/tourist_hotel_country_freq/diff/'):
            os.makedirs('resources/bow/tourist_hotel_country_freq/diff/')
        with open('resources/bow/tourist_hotel_country_freq/diff/' + keyword + '.csv',
                  mode='w') as file:
            writer = csv.writer(file, delimiter='|', quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Tourist_Country_Index', 'Tourist_Country', 'Hotel_Country_Index', 'Hotel_Country',
                             'Total number of unique reviews', 'Token_Index', 'Token', 'Token_Frequence_in_Good',
                             'Token_Frequence_in_Bad', 'Difference'])
            for countries in diff_tables[keyword].keys():
                for tok in diff_tables[keyword][countries]['tokens'].keys():
                    writer.writerow([country_tourist_ind['country_to_index'][countries[0]], countries[0],
                                     country_hotel_ind['country_to_index'][countries[1]], countries[1],
                                     diff_tables[keyword][countries]['count_rev'], token_index['token_to_index'][tok], tok,
                                     "{:.15f}".format(diff_tables[keyword][countries]['tokens'][tok]['good']),
                                     "{:.15f}".format(diff_tables[keyword][countries]['tokens'][tok]['bad']),
                                     "{:.15f}".format(diff_tables[keyword][countries]['tokens'][tok]['diff'])])
        file.close()
        print(str(time.time() - start_time) + ' seconds to write the difference matrix for ' + keyword)