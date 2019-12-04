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
        #country_tourist_ind = indexmanager.get_tourist_country_index()
        country_ind = indexmanager.get_hotel_country_index()
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
                    writer.writerow([country_ind['country_to_index'][countries[0]], countries[0],
                                     country_ind['country_to_index'][countries[1]], countries[1],
                                     diff_tables[keyword][countries]['count_rev'], token_index['token_to_index'][tok], tok,
                                     "{:.15f}".format(diff_tables[keyword][countries]['tokens'][tok]['good']),
                                     "{:.15f}".format(diff_tables[keyword][countries]['tokens'][tok]['bad']),
                                     "{:.15f}".format(diff_tables[keyword][countries]['tokens'][tok]['diff'])])

        file.close()
        print(str(time.time() - start_time) + ' seconds to write the difference matrix for ' + keyword)


def filter(originfile):
    keywords = helper.getKeywords(originfile)
    countries=dict()
    tokens=dict()
    countries['origin']=dict()
    countries['destination']=dict()
    lines_dict=dict()
    intersect_tokens=set()
    intersect_countries_origin=set()
    intersect_countries_dest = set()
    validkeywords = []
    ass_count_count=dict()
    ass_count_count['origin_tourist'] = dict()
    ass_count_count['destination_hotel'] = dict()
    combs=dict()
    for keyword in list(keywords.keys()):
        if keyword in ['breakfast', 'bedroom', 'bathroom' ,'location']:
            combs[keyword]=set()
            ass_count_count['origin_tourist'][keyword]=dict()
            ass_count_count['destination_hotel'][keyword] = dict()
            countries['origin'][keyword] = set()
            countries['destination'][keyword] = set()
            tokens[keyword]=set()
            start_time = time.time()
            goforward = True
            print(keyword + ' ---- ' + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
            lines=[]
            try:
                with open('resources/bow/tourist_hotel_country_freq/diff/' + keyword + '.csv') as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter='|')
                    next(csv_reader)
                    for row in csv_reader:
                        if int(row[4])>=100 and row[1]!='' and row[1]!='no_country' and row[3]!='no_country':
                            combs[keyword].add((row[0],row[2]))
                            lines.append([row[0],row[2],row[5],row[9]])
                            countries['origin'][keyword].add(row[0])
                            countries['destination'][keyword].add(row[2])
                            tokens[keyword].add(row[5])
                csv_file.close()
            except Exception as e:
                goforward=False
            if goforward:
                validkeywords.append(keyword)
                lines_dict[keyword]=lines
                if len(list(intersect_tokens))==0:
                    intersect_tokens=tokens[keyword]
                if len(list(intersect_countries_origin))==0:
                    intersect_countries_origin=countries['origin'][keyword]
                if len(list(intersect_countries_dest))==0:
                    intersect_countries_dest=countries['destination'][keyword]
                intersect_tokens=intersect_tokens.intersection(tokens[keyword])
                intersect_countries_origin=intersect_countries_origin.intersection(countries['origin'][keyword])
                intersect_countries_dest = intersect_countries_dest.intersection(countries['destination'][keyword])
            print('------------------------------------------------------')
            print(str(time.time() - start_time) + ' seconds to filter ' + keyword)
    if not os.path.exists('resources/bow/tourist_hotel_country_freq/diff/filtered/'):
        os.makedirs('resources/bow/tourist_hotel_country_freq/diff/filtered/')
    if not os.path.exists('resources/bow/tourist_hotel_country_freq/diff/filtered/withcomb/'):
        os.makedirs('resources/bow/tourist_hotel_country_freq/diff/filtered/withcomb/')
    combs_intersection = set.intersection(*[combs[keyword] for keyword in combs.keys()])
    lines=[[line for line in lines_dict[keyword] if (line[0],line[1]) in combs_intersection] for keyword in validkeywords]
    ass = dict()
    common_tokens=dict()
    for line in lines:
        ass[lines.index(line)] = dict()
        for l in line:
            if (l[0],l[1]) not in common_tokens.keys():
                common_tokens[(l[0],l[1])]=set()
            common_tokens[(l[0], l[1])].add(l[2])
            if l[0] not in ass[lines.index(line)].keys():
                ass[lines.index(line)][l[0]] = set()
            ass[lines.index(line)][l[0]].add(l[1])
    origins = set.intersection(
        *[set([key for key in ass[keyword].keys()
               if len(ass[keyword][key]) >= 6])
          for keyword in ass.keys()])
    destinations = set.intersection(
        *[set.intersection(*[ass[keyword][key]
                             for key in ass[keyword].keys()
                             if len(ass[keyword][key]) >= 6]) for
          keyword in ass.keys()])
    token_index=dict()
    country_index=dict()
    old_cont_index=indexmanager.get_hotel_country_index()
    old_tok_index = indexmanager.get_token_index()
    country_list=list(destinations.union(origins))
    token_list=list(set.union(*[common_tokens[k] for k in common_tokens.keys()]))
    for i in range(1,len(country_list)+1):
        country_index[i]=old_cont_index['index_to_country'][int(country_list[i-1])]
    for i in range(1,len(token_list)):
        token_index[i]=old_tok_index['index_to_token'][int(token_list[i-1])]
    with open('resources/bow/tourist_hotel_country_freq/diff/filtered/withcomb/token_index.csv', mode='w') as file:
        writer = csv.writer(file, delimiter='|', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        for k in sorted(list(token_index.keys())):
            writer.writerow([k, token_index[k]])
    file.close()
    with open('resources/bow/tourist_hotel_country_freq/diff/filtered/withcomb/country_index.csv', mode='w') as file:
        writer = csv.writer(file, delimiter='|', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        for k in sorted(list(country_index.keys())):
            writer.writerow([k, country_index[k]])
    file.close()
    for keyword in validkeywords:
        with open('resources/bow/tourist_hotel_country_freq/diff/filtered/' + keyword + '.csv',
                  mode='w') as file:
            writer = csv.writer(file, delimiter='|', quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['country_origin_index','country_destination_index','token_index','frequence_difference'])
            for line in lines_dict[keyword]:
                if line[0] in intersect_countries_origin and line[1] in intersect_countries_dest:
                    writer.writerow(line)
        file.close()
        with open('resources/bow/tourist_hotel_country_freq/diff/filtered/withcomb/' + keyword + '.csv',
                  mode='w') as file:
            writer = csv.writer(file, delimiter='|', quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['country_origin_index','country_destination_index','token_index','frequence_difference'])
            for line in lines_dict[keyword]:
                if line[0] in origins and line[1] in destinations:
                    writer.writerow(line)
        file.close()
def build_association_count_list(originfile):
    lines=[]
    ass = dict()
    keywords = helper.getKeywords(originfile)
    combs=dict()
    for keyword in list(keywords.keys()):
        if keyword in ['breakfast', 'bedroom', 'bathroom', 'location']:
            ass[keyword] = dict()
            combs[keyword]=dict()
            with open(
                    'resources/bow/tourist_hotel_country_freq/diff/filtered/withcomb/' + keyword + '.csv') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter='|')
                next(csv_reader)
                for row in csv_reader:
                    if row[0] not in ass[keyword].keys():
                        ass[keyword][row[0]] = set()
                    ass[keyword][row[0]].add(row[1])
                    if (row[0], row[1]) not in combs[keyword].keys():
                        combs[keyword][(row[0], row[1])]=set()
                    combs[keyword][(row[0], row[1])].add(row[2])
    b = True
    v = set([[k for k in ass[keyword].keys()]for keyword in ass.keys()][0])
    for o in [[k for k in ass[keyword].keys()]for keyword in ass.keys()]:
        if set(o)!=v:
            b=False
            break
    lines.append('all possible origins are the same number and the same = '+str(b)+'\n')
    v = [[ass[keyword][k] for k in ass[keyword].keys()] for keyword in ass.keys()][0][0]
    b = True
    for d in [[ass[keyword][k] for k in ass[keyword].keys()] for keyword in ass.keys()]:
        for dd in d:
            if dd != v:
                b = False
                break
    lines.append('all possible destinations are the same number and the same = '+str(b)+'\n')
    lines.append('all origins are: '+str(set([k for k in ass['breakfast'].keys()]))+'\n')
    lines.append('all destinations are: ' + str([ass['breakfast'][k] for k in ass['breakfast'].keys()][0])+'\n')
    for keyword in ['breakfast', 'bedroom', 'bathroom', 'location']:
        b = True
        toksetz=[combs[keyword][c] for c in combs[keyword].keys()][0]
        for tokset in [combs[keyword][c] for c in combs[keyword].keys()]:
            if tokset!=toksetz:
                b=False
                break
        lines.append("for concept "+ keyword+', for every combination origin/destination, all the tokens are the same = '+str(b)+'\n')
        lines.append("for concept "+ keyword+' the length of the list of tokens for the first combination origin/destination is '+str(len(toksetz))+'\n')
    file = open('resources/bow/tourist_hotel_country_freq/diff/filtered/withcomb/report.txt','w')
    file.writelines(lines)
    file.close()