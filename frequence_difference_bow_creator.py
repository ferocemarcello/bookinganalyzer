import csv
import os
import time
import operator
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


def filter(originfile):
    keywords = helper.getKeywords(originfile)
    for keyword in list(keywords.keys()):
        start_time = time.time()
        goforward = True
        validkeywords = []
        print(keyword + ' ---- ' + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
        lines=[]
        try:
            with open('resources/bow/tourist_hotel_country_freq/diff/' + keyword + '.csv') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter='|')
                lines.append(next(csv_reader))
                for row in csv_reader:
                    if int(row[4])>=100 and row[1]!='' and row[3]!='no_location_of_hotel':
                        lines.append(row)
            csv_file.close()
            if not os.path.exists('resources/bow/tourist_hotel_country_freq/diff/filtered/'):
                os.makedirs('resources/bow/tourist_hotel_country_freq/diff/filtered/')
            with open('resources/bow/tourist_hotel_country_freq/diff/filtered/' + keyword + '.csv',
                      mode='w') as file:
                writer = csv.writer(file, delimiter='|', quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)
                writer.writerows(lines)
            file.close()
        except:
            goforward=False
        if goforward:
            validkeywords.append(keyword)
        print('------------------------------------------------------')
        print(str(time.time() - start_time) + ' seconds to filter ' + keyword)
def build_association_count_list(originfile):
    keywords = helper.getKeywords(originfile)
    ass_count_count=dict()
    valid_keywords = []
    ass_count_count['origin_tourist']=dict()
    ass_count_count['destination_hotel']=dict()
    for keyword in list(keywords.keys()):
        start_time = time.time()
        print(keyword + ' ---- ' + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
        ass_count_count['origin_tourist'][keyword] = dict()
        ass_count_count['destination_hotel'][keyword] = dict()
        goforward = True
        try:
            with open('resources/bow/tourist_hotel_country_freq/diff/filtered/' + keyword + '.csv') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter='|')
                next(csv_reader)
                for row in csv_reader:
                    if row[1] not in ass_count_count['origin_tourist'][keyword].keys():
                        ass_count_count['origin_tourist'][keyword][row[1]]=dict()
                    if row[3] not in ass_count_count['destination_hotel'][keyword].keys():
                        ass_count_count['destination_hotel'][keyword][row[3]] = dict()
                    if row[3] not in ass_count_count['origin_tourist'][keyword][row[1]].keys():
                        ass_count_count['origin_tourist'][keyword][row[1]][row[3]]=row[4]
                    if row[1] not in ass_count_count['destination_hotel'][keyword][row[3]].keys():
                        ass_count_count['destination_hotel'][keyword][row[3]][row[1]]=row[4]
            csv_file.close()
        except:
            goforward = False
        if goforward:
            valid_keywords.append(keyword)
        print('------------------------------------------------------')
        print(str(time.time() - start_time) + ' seconds for creating dicts for association count for ' + keyword)
    allorigins = set()
    alldestinations=set()
    for keyword in valid_keywords:
        allorigins=allorigins.union(set(ass_count_count['origin_tourist'][keyword].keys()))
        alldestinations=alldestinations.union(set(ass_count_count['destination_hotel'][keyword].keys()))
    s = dict()
    s['origin']=dict()
    s['destination']=dict()
    for origin in allorigins:
        s['origin'][origin]=0
        for keyword in valid_keywords:
            if origin not in ass_count_count['origin_tourist'][keyword].keys():
                ass_count_count['origin_tourist'][keyword][origin]=dict()
    for dest in alldestinations:
        s['destination'][dest] = 0
        for keyword in valid_keywords:
            if dest not in ass_count_count['destination_hotel'][keyword].keys():
                ass_count_count['destination_hotel'][keyword][dest]=dict()
    count_sum_dict=dict()
    for keyword in valid_keywords:
        count_sum_dict[keyword]=dict()
        count_sum_dict[keyword]['origin']= dict()
        count_sum_dict[keyword]['destination'] = dict()
        for origin in allorigins:
            list_from_origin_keyword=[int(ass_count_count['origin_tourist'][keyword][origin][x]) for x in
             ass_count_count['origin_tourist'][keyword][origin].keys()]
            sumfromorigin_keyword=sum(list_from_origin_keyword)
            num_distinct_from_origin_keyword = len(list_from_origin_keyword)
            count_sum_dict[keyword]['origin'][origin] ={"count distinct country destinations":num_distinct_from_origin_keyword,"sum of unique reviews from this country":sumfromorigin_keyword}
        for dest in alldestinations:
            list_to_dest_keyword=[int(ass_count_count['destination_hotel'][keyword][dest][x]) for x in
             ass_count_count['destination_hotel'][keyword][dest].keys()]
            sum_to_dest_keyword=sum(list_to_dest_keyword)
            num_distinct_to_destination_keyword = len(list_to_dest_keyword)
            count_sum_dict[keyword]['destination'][dest]={"count distinct country of origin":num_distinct_to_destination_keyword,"sum of unique reviews to the country":sum_to_dest_keyword}
    for k in count_sum_dict:
        for origin in allorigins:
            s['origin'][origin]+=count_sum_dict[k]['origin'][origin]['count distinct country destinations']
        for dest in alldestinations:
            s['destination'][dest] += count_sum_dict[k]['destination'][dest]['count distinct country of origin']
    sorted_origins=[x[0] for x in sorted(s['origin'].items(), key=operator.itemgetter(1), reverse=True)]
    sorted_destinations=[x[0] for x in sorted(s['destination'].items(), key=operator.itemgetter(1),reverse=True)]
    line=['']+sorted_origins+['']+sorted_destinations
    lines=[]
    lines.append(line)
    for keyword in valid_keywords:
        line = [keyword]
        for origin in sorted_origins:
            line.append(count_sum_dict[keyword]['origin'][origin])
        line.append("")
        for dest in sorted_destinations:
            line.append(count_sum_dict[keyword]['destination'][dest])
        lines.append(line)
    with open('resources/bow/tourist_hotel_country_freq/diff/filtered/association_count.csv', mode='w') as file:
        writer = csv.writer(file, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(lines)
    file.close()