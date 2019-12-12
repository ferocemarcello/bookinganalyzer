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


def get_diff_table(good_table, bad_table, tokenset,common_tokens=True):
    diff_table={}
    if common_tokens:
        for countries in bad_table.keys():
            if countries in good_table.keys():
                diff_table[countries] = {}
                diff_table[countries]['tokens'] = {}
                diff_table[countries]['unique_reviews']=bad_table[countries]['unique_reviews'].union(good_table[countries]['unique_reviews'])
                diff_table[countries]['count_rev']=len(list(diff_table[countries]['unique_reviews']))
                for tok in bad_table[countries]['tokens'].keys():
                    if tok in good_table[countries]['tokens'].keys():
                        tokenset.add(tok)
                        diff_table[countries]['tokens'][tok] = {}
                        diff_table[countries]['tokens'][tok]['good'] = good_table[countries]['tokens'][tok]
                        diff_table[countries]['tokens'][tok]['bad'] = bad_table[countries]['tokens'][tok]
                        diff_table[countries]['tokens'][tok]['diff'] = good_table[countries]['tokens'][tok]-bad_table[countries]['tokens'][tok]
    else:
        allcountries = set(good_table.keys()).union(set(bad_table.keys()))
        for countries in allcountries:
            diff_table[countries]={}
            diff_table[countries]['tokens'] = {}
            diff_table[countries]['unique_reviews']=set()
            if countries in bad_table.keys() and countries in good_table.keys():
                diff_table[countries]['unique_reviews']=bad_table[countries]['unique_reviews'].union(good_table[countries]['unique_reviews'])
                alltokens=set(good_table[countries]['tokens'].keys()).union(set(bad_table[countries]['tokens'].keys()))
            elif countries in good_table.keys():
                diff_table[countries]['unique_reviews']=good_table[countries]['unique_reviews']
                alltokens = set(good_table[countries]['tokens'].keys())
            elif countries in bad_table.keys():
                diff_table[countries]['unique_reviews'] = bad_table[countries]['unique_reviews']
                alltokens = set(bad_table[countries]['tokens'].keys())
            diff_table[countries]['count_rev']=len(list(diff_table[countries]['unique_reviews']))
            for tok in alltokens:
                tokenset.add(tok)
                diff_table[countries]['tokens'][tok]={}
                if countries in bad_table.keys() and tok in bad_table[countries]['tokens'].keys() and countries in good_table.keys() and tok in good_table[countries]['tokens'].keys():
                    diff_table[countries]['tokens'][tok]['good'] = good_table[countries]['tokens'][tok]
                    diff_table[countries]['tokens'][tok]['bad'] = bad_table[countries]['tokens'][tok]
                    diff_table[countries]['tokens'][tok]['diff'] = good_table[countries]['tokens'][tok]-bad_table[countries]['tokens'][tok]
                elif countries in good_table.keys() and tok in good_table[countries]['tokens'].keys():
                    diff_table[countries]['tokens'][tok]['good'] = good_table[countries]['tokens'][tok]
                    diff_table[countries]['tokens'][tok]['bad'] = 'N/A'
                    diff_table[countries]['tokens'][tok]['diff'] = good_table[countries]['tokens'][tok]
                elif countries in bad_table.keys() and tok in bad_table[countries]['tokens'].keys():
                    diff_table[countries]['tokens'][tok]['bad'] = bad_table[countries]['tokens'][tok]
                    diff_table[countries]['tokens'][tok]['good'] = 'N/A'
                    diff_table[countries]['tokens'][tok]['diff'] = 0.0-bad_table[countries]['tokens'][tok]
    return diff_table


def do(originfile, all=False, common_tokens=True):
    if all:
        tokenset=set()
        alltable=read_table('resources/bow/tourist_hotel_country_freq/all.csv')
        diff_table = {}
        for countries in alltable.keys():
            diff_table[countries] = {}
            diff_table[countries]['tokens'] = {}
            diff_table[countries]['unique_reviews'] = alltable[countries]['unique_reviews']
            diff_table[countries]['count_rev'] = len(list(diff_table[countries]['unique_reviews']))
            for tok in alltable[countries]['tokens'].keys():
                tokenset.add(tok)
                diff_table[countries]['tokens'][tok] = {}
                diff_table[countries]['tokens'][tok]['diff'] = alltable[countries]['tokens'][tok]
        indexmanager.update_token_index(tokenset)
        print("start writing difference matrix for all matrix")
        country_ind = indexmanager.get_hotel_country_index()
        if not os.path.exists('resources/bow/tourist_hotel_country_freq/diff/'):
            os.makedirs('resources/bow/tourist_hotel_country_freq/diff/')
        with open('resources/bow/tourist_hotel_country_freq/diff/all.csv',
                  mode='w') as file:
            writer = csv.writer(file, delimiter='|', quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Tourist_Country_Index', 'Tourist_Country', 'Hotel_Country_Index', 'Hotel_Country',
                             'Total number of unique reviews', 'Token_Index', 'Token', 'Token_Frequence'])
            token_index = indexmanager.get_token_index()
            for countries in diff_table.keys():
                for tok in diff_table[countries]['tokens'].keys():
                    writer.writerow([country_ind['country_to_index'][countries[0]], countries[0],
                                     country_ind['country_to_index'][countries[1]], countries[1],
                                     diff_table[countries]['count_rev'], token_index['token_to_index'][tok],
                                     tok,
                                     "{:.15f}".format(diff_table[countries]['tokens'][tok]['diff'])])

        file.close()
    else:
        keywords = helper.getKeywords(originfile)
        tokenset = set()
        diff_tables={}
        validkeywords=[]
        for keyword in keywords.keys():
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
                diff_table=get_diff_table(good_tab,bad_table,tokenset,common_tokens=common_tokens)
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
                        goodval=diff_tables[keyword][countries]['tokens'][tok]['good']
                        if goodval!='N/A':
                            goodval="{:.15f}".format(goodval)
                        badval = diff_tables[keyword][countries]['tokens'][tok]['bad']
                        if badval != 'N/A':
                            badval = "{:.15f}".format(badval)
                        writer.writerow([country_ind['country_to_index'][countries[0]], countries[0],
                                         country_ind['country_to_index'][countries[1]], countries[1],
                                         diff_tables[keyword][countries]['count_rev'], token_index['token_to_index'][tok], tok,
                                         goodval,
                                         badval,
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
    lines_reduced_dict = dict()
    intersect_tokens=set()
    intersect_countries_origin=set()
    intersect_countries_dest = set()
    validkeywords = []
    ass_count_count=dict()
    ass_count_count['origin_tourist'] = dict()
    ass_count_count['destination_hotel'] = dict()
    k_values=dict()
    k_values['breakfast']=6
    k_values['bedroom']=5
    k_values['bathroom']=4
    k_values['location']=13
    for keyword in list(keywords.keys()):
        if keyword in ['breakfast', 'bedroom', 'bathroom' ,'location']:
            ass_count_count['origin_tourist'][keyword]=dict()
            ass_count_count['destination_hotel'][keyword] = dict()
            countries['origin'][keyword] = set()
            countries['destination'][keyword] = set()
            tokens[keyword]=set()
            start_time = time.time()
            goforward = True
            print(keyword + ' ---- ' + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
            lines=[]
            lines_reduced=[]
            try:
                with open('resources/bow/tourist_hotel_country_freq/diff/' + keyword + '.csv') as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter='|')
                    next(csv_reader)
                    for row in csv_reader:
                        if int(row[4])>=100 and row[1]!='' and row[1]!='no_country' and row[3]!='no_country':
                            lines.append([row[0],row[2],row[5],row[9]])
                            countries['origin'][keyword].add(row[0])
                            countries['destination'][keyword].add(row[2])
                            tokens[keyword].add(row[5])
                        if int(row[4])>=20 and row[1]!='' and row[1]!='no_country' and row[3]!='no_country':
                            lines_reduced.append([row[0],row[2],row[5],row[9]])
                csv_file.close()
            except Exception as e:
                goforward=False
            if goforward:
                validkeywords.append(keyword)
                lines_dict[keyword]=lines
                lines_reduced_dict[keyword]=lines_reduced
                if len(list(intersect_tokens))==0:
                    intersect_tokens=tokens[keyword]
                if len(list(intersect_countries_origin))==0:
                    intersect_countries_origin=countries['origin'][keyword]
                if len(list(intersect_countries_dest))==0:
                    intersect_countries_dest=countries['destination'][keyword]
                intersect_tokens=intersect_tokens.intersection(tokens[keyword])
                intersect_countries_origin=intersect_countries_origin.intersection(countries['origin'][keyword])
                intersect_countries_dest = intersect_countries_dest.intersection(countries['destination'][keyword])

                ass_sep=dict()
                for line in lines:
                    if line[0] not in ass_sep.keys():
                        ass_sep[line[0]] = set()
                    ass_sep[line[0]].add(line[1])

                k=k_values[keyword]
                destinations_sep=set.intersection(*[ass_sep[key] for key in ass_sep.keys() if len(ass_sep[key]) >= k])
                origins_sep = set([key for key in ass_sep.keys() if
                                                  ass_sep[key] >= (destinations_sep)])
                newdestinations_sep = set.intersection(*[ass_sep[k] for k in origins_sep])
                if not os.path.exists(
                        'resources/bow/tourist_hotel_country_freq/diff/filtered/withcomb/concept_separetely/'):
                    os.makedirs('resources/bow/tourist_hotel_country_freq/diff/filtered/withcomb/concept_separetely/')
                with open('resources/bow/tourist_hotel_country_freq/diff/filtered/withcomb/concept_separetely/' + keyword + '.csv',
                          mode='w') as file:
                    writer = csv.writer(file, delimiter='|', quotechar='"',
                                        quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(
                        ['country_origin_index', 'country_destination_index', 'token_index', 'frequence_difference'])
                    for line in lines_dict[keyword]:
                        if line[0] in origins_sep and line[1] in newdestinations_sep:
                            writer.writerow(line)
                file.close()
            print('------------------------------------------------------')
            print(str(time.time() - start_time) + ' seconds to filter ' + keyword)
    if not os.path.exists('resources/bow/tourist_hotel_country_freq/diff/filtered/'):
        os.makedirs('resources/bow/tourist_hotel_country_freq/diff/filtered/')
    if not os.path.exists('resources/bow/tourist_hotel_country_freq/diff/filtered/withcomb/'):
        os.makedirs('resources/bow/tourist_hotel_country_freq/diff/filtered/withcomb/')
    if not os.path.exists('resources/bow/tourist_hotel_country_freq/diff/filtered/withcomb/reduced/'):
        os.makedirs('resources/bow/tourist_hotel_country_freq/diff/filtered/withcomb/reduced/')
    lines = [[line for line in lines_dict[keyword]] for keyword in
             validkeywords]
    lines_reduced = [[line for line in lines_reduced_dict[keyword]] for keyword in
             validkeywords]
    ass = dict()
    ass_reduced=dict()
    tokens=set()
    for line in lines:
        ass[lines.index(line)] = dict()
        for l in line:
            if l[0] not in ass[lines.index(line)].keys():
                ass[lines.index(line)][l[0]] = set()
            ass[lines.index(line)][l[0]].add(l[1])
    tokens_reduced=set()
    for line in lines_reduced:
        ass_reduced[lines_reduced.index(line)] = dict()
        for l in line:
            if l[0] not in ass_reduced[lines_reduced.index(line)].keys():
                ass_reduced[lines_reduced.index(line)][l[0]] = set()
            ass_reduced[lines_reduced.index(line)][l[0]].add(l[1])
    k = 7
    destinations = set.intersection(
        *[set.intersection(*[ass[keyword][key]
                             for key in ass[keyword].keys()
                             if len(ass[keyword][key]) >= k]) for
          keyword in ass.keys()])
    origins = set.intersection(*[set([key for key in ass[keyword].keys() if
                                       ass[keyword][key]>=(destinations)]) for keyword in
                                  ass.keys()])
    newdestinations=set.intersection(*[set.intersection(*[ass[keyword][k] for k in origins]) for keyword in ass.keys()])
    for keyword in lines_dict.keys():
        for line in lines_dict[keyword]:
            if line[0] in origins and line[1] in newdestinations:
                tokens.add(line[2])
    k=12
    destinations_reduced = set.intersection(
        *[set.intersection(*[ass_reduced[keyword][key]
                             for key in ass_reduced[keyword].keys()
                             if len(ass_reduced[keyword][key]) >= k]) for
          keyword in ass_reduced.keys()])
    origins_reduced = set.intersection(*[set([key for key in ass_reduced[keyword].keys() if
                                      ass_reduced[keyword][key] >= (destinations_reduced)]) for keyword in
                                 ass_reduced.keys()])
    newdestinations_reduced = set.intersection(
        *[set.intersection(*[ass_reduced[keyword][k] for k in origins_reduced]) for keyword in ass.keys()])
    for keyword in lines_reduced_dict.keys():
        for line in lines_reduced_dict[keyword]:
            if line[0] in origins_reduced and line[1] in newdestinations_reduced:
                tokens_reduced.add(line[2])
    token_index=dict()
    country_index=dict()
    token_index_reduced = dict()
    country_index_reduced = dict()
    old_cont_index=indexmanager.get_hotel_country_index()
    old_tok_index = indexmanager.get_token_index()
    country_list=list(newdestinations.union(origins))
    old_cont_to_new=dict()
    old_tok_to_new= dict()
    old_cont_to_new_reduced = dict()
    old_tok_to_new_reduced = dict()
    tokenlist=list(tokens)
    tokenlist_reduced=list(tokens_reduced)
    country_list_reduced=list(newdestinations_reduced.union(origins_reduced))
    for i in range(1,len(country_list)+1):
        country_index[i]=old_cont_index['index_to_country'][int(country_list[i-1])]
        old_cont_to_new[int(country_list[i-1])]=i
    for i in range(1,len(tokenlist)+1):
        token_index[i]=old_tok_index['index_to_token'][int(tokenlist[i-1])]
        old_tok_to_new[int(tokenlist[i-1])]=i
    for i in range(1,len(country_list_reduced)+1):
        country_index_reduced[i]=old_cont_index['index_to_country'][int(country_list_reduced[i-1])]
        old_cont_to_new_reduced[int(country_list_reduced[i-1])]=i
    for i in range(1,len(tokenlist)+1):
        token_index_reduced[i]=old_tok_index['index_to_token'][int(tokenlist_reduced[i-1])]
        old_tok_to_new_reduced[int(tokenlist_reduced[i-1])]=i
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
                if line[0] in origins and line[1] in newdestinations:
                    newline=[old_cont_to_new[int(line[0])],old_cont_to_new[int(line[1])],old_tok_to_new[int(line[2])],line[3]]
                    writer.writerow(newline)
        with open('resources/bow/tourist_hotel_country_freq/diff/filtered/withcomb/reduced/' + keyword + '.csv',
                  mode='w') as file:
            writer = csv.writer(file, delimiter='|', quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['country_origin_index','country_destination_index','token_index','frequence_difference'])
            for line in lines_reduced_dict[keyword]:
                if line[0] in origins_reduced and line[1] in newdestinations_reduced:
                    writer.writerow(line)
        file.close()
def build_association_count_list(originfile):
    lines=[]
    lines_reduced=[]
    ass = dict()
    keywords = helper.getKeywords(originfile)
    combs=dict()
    ass_reduced=dict()
    combs_reduced=dict()
    for keyword in list(keywords.keys()):
        if keyword in ['breakfast', 'bedroom', 'bathroom', 'location']:
            ass[keyword] = dict()
            combs[keyword]=dict()
            ass_reduced[keyword] = dict()
            combs_reduced[keyword] = dict()
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
            csv_file.close()
            with open(
                    'resources/bow/tourist_hotel_country_freq/diff/filtered/withcomb/reduced/' + keyword + '.csv') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter='|')
                next(csv_reader)
                for row in csv_reader:
                    if row[0] not in ass_reduced[keyword].keys():
                        ass_reduced[keyword][row[0]] = set()
                    ass_reduced[keyword][row[0]].add(row[1])
                    if (row[0], row[1]) not in combs_reduced[keyword].keys():
                        combs_reduced[keyword][(row[0], row[1])]=set()
                    combs_reduced[keyword][(row[0], row[1])].add(row[2])
            csv_file.close()
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

    b = True
    v = set([[k for k in ass_reduced[keyword].keys()] for keyword in ass_reduced.keys()][0])
    for o in [[k for k in ass_reduced[keyword].keys()] for keyword in ass_reduced.keys()]:
        if set(o) != v:
            b = False
            break
    lines_reduced.append('all possible origins are the same number and the same = ' + str(b) + '\n')
    v = [[ass_reduced[keyword][k] for k in ass_reduced[keyword].keys()] for keyword in ass_reduced.keys()][0][0]
    b = True
    for d in [[ass_reduced[keyword][k] for k in ass_reduced[keyword].keys()] for keyword in ass_reduced.keys()]:
        for dd in d:
            if dd != v:
                b = False
                break
    lines_reduced.append('all possible destinations are the same number and the same = ' + str(b) + '\n')
    lines_reduced.append('all origins are: ' + str(set([k for k in ass_reduced['breakfast'].keys()])) + '\n')
    lines_reduced.append('all destinations are: ' + str([ass_reduced['breakfast'][k] for k in ass_reduced['breakfast'].keys()][0]) + '\n')
    for keyword in ['breakfast', 'bedroom', 'bathroom', 'location']:
        b = True
        toksetz = [combs_reduced[keyword][c] for c in combs_reduced[keyword].keys()][0]
        for tokset in [combs_reduced[keyword][c] for c in combs_reduced[keyword].keys()]:
            if tokset != toksetz:
                b = False
                break
        lines_reduced.append(
            "for concept " + keyword + ', for every combination origin/destination, all the tokens are the same = ' + str(
                b) + '\n')
        lines_reduced.append(
            "for concept " + keyword + ' the length of the list of tokens for the first combination origin/destination is ' + str(
                len(toksetz)) + '\n')

    file = open('resources/bow/tourist_hotel_country_freq/diff/filtered/withcomb/report.txt','w')
    file.writelines(lines)
    file.close()
    file = open('resources/bow/tourist_hotel_country_freq/diff/filtered/withcomb/reduced/report.txt', 'w')
    file.writelines(lines_reduced)
    file.close()

def filterallsep(originfile):
    keywords = helper.getKeywords(originfile)
    old_cont_index = indexmanager.get_hotel_country_index()
    old_tok_index = indexmanager.get_token_index()
    for keyword in list(keywords.keys())+['all']:
        try:
            combs=dict()
            origins_to_dect = dict()
            goforward=True
            start_time = time.time()
            lines = []
            tokens = set()
            if keyword=='all':
                frequencecell=7
            else:
                frequencecell=9
            with open('resources/bow/tourist_hotel_country_freq/diff/' + keyword + '.csv') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter='|')
                header=next(csv_reader)
                for row in csv_reader:
                    if int(row[4]) >= 0 and row[1] != '' and row[1] != 'no_country' and row[3] != 'no_country':
                        lines.append([row[0], row[2], row[4], row[5], row[frequencecell]])
                        if (row[0], row[2]) not in combs.keys():
                            combs[(row[0], row[2])]=set()
                        if row[0] not in origins_to_dect.keys():
                            origins_to_dect[row[0]]=set()
                        origins_to_dect[row[0]].add(row[2])
                        combs[(row[0], row[2])].add(row[5])
                        tokens.add(row[5])
            #if len(origins_to_dect[x])>=5
            #print(len([x for x in origins_to_dect.keys()]))
            csv_file.close()
        except Exception as e:
            goforward = False
        if goforward:
            countries=set(origins_to_dect.keys()).union(set.union(*[x for x in origins_to_dect.values()]))
            countries = list(countries)
            country_index=dict()
            old_cont_to_new=dict()
            for i in range(1,len(countries)+1):
                country_index[i] = old_cont_index['index_to_country'][int(countries[i - 1])]
                old_cont_to_new[int(countries[i - 1])] = i
            tokens = list(tokens)
            token_index = dict()
            old_tok_to_new = dict()
            for i in range(1, len(tokens) + 1):
                token_index[i] = old_tok_index['index_to_token'][int(tokens[i - 1])]
                old_tok_to_new[int(tokens[i - 1])] = i
            if not os.path.exists(
                    'resources/bow/tourist_hotel_country_freq/diff/filtered/all_separetely/'):
                os.makedirs('resources/bow/tourist_hotel_country_freq/diff/filtered/all_separetely/')
            with open(
                    'resources/bow/tourist_hotel_country_freq/diff/filtered/all_separetely/' + keyword + '.csv',
                    mode='w') as file:
                writer = csv.writer(file, delimiter='|', quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)
                writer.writerow(
                    ['country_origin_index', 'country_destination_index', 'number unique reviews', 'token_index', 'frequence_difference'])
                for line in lines:
                    newline = [old_cont_to_new[int(line[0])], old_cont_to_new[int(line[1])],int(line[2]),
                               old_tok_to_new[int(line[3])], line[4]]
                    writer.writerow(newline)
            file.close()
            with open(
                    'resources/bow/tourist_hotel_country_freq/diff/filtered/all_separetely/' + keyword + '_country_index.csv',
                    mode='w') as file:
                writer = csv.writer(file, delimiter='|', quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)
                writer.writerow(
                    ['country_index', 'country'])
                for key in country_index.keys():
                    writer.writerow([key,country_index[key]])
            file.close()
            with open(
                    'resources/bow/tourist_hotel_country_freq/diff/filtered/all_separetely/' + keyword + '_token_index.csv',
                    mode='w') as file:
                writer = csv.writer(file, delimiter='|', quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)
                writer.writerow(
                    ['token_index', 'token'])
                for key in token_index.keys():
                    writer.writerow([key,token_index[key]])
            file.close()
        print('------------------------------------------------------')
        print(str(time.time() - start_time) + ' seconds to filter ' + keyword)