import csv
import helper
import numpy as np

destinations=dict()
origins=dict()
keywords = helper.getKeywords('booking_keywords.txt')
for keyword in list(keywords.keys()):
    if keyword in ['breakfast', 'bedroom', 'bathroom', 'location']:
        destinations[keyword]=set()
        origins[keyword] = set()
        with open('resources/bow/tourist_hotel_country_freq/diff/filtered/withcomb/' + keyword + '.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='|')
            next(csv_reader)
            for row in csv_reader:
                origins[keyword].add(row[0])
                destinations[keyword].add(row[1])
print(origins['breakfast']==origins['bedroom']==origins['bathroom']==origins['location'])
unorigin=((origins['breakfast'].union(origins['bedroom'])).union(origins['bathroom'])).union(origins['location'])
print(destinations['breakfast']==destinations['bedroom']==destinations['bathroom']==destinations['location'])
undest=((destinations['breakfast'].union(destinations['bedroom'])).union(destinations['bathroom'])).union(destinations['location'])
for keyword in list(keywords.keys()):
    if keyword in ['breakfast', 'bedroom', 'bathroom', 'location']:
        print(unorigin.difference(origins[keyword]))
        print(undest.difference(destinations[keyword]))
matrices=dict()
matrices['breakfast'] = np.zeros((len(list(origins['breakfast'])), len(list(destinations['breakfast']))))
matrices['bedroom'] = np.zeros((len(list(origins['bedroom'])), len(list(destinations['bedroom']))))
matrices['bathroom'] = np.zeros((len(list(origins['bathroom'])), len(list(destinations['bathroom']))))
matrices['location'] = np.zeros((len(list(origins['location'])), len(list(destinations['location']))))
ass=dict()
for keyword in list(keywords.keys()):
    if keyword in ['breakfast', 'bedroom', 'bathroom', 'location']:
        ass[keyword]=dict()
        with open('resources/bow/tourist_hotel_country_freq/diff/filtered/withcomb/' + keyword + '.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='|')
            next(csv_reader)
            for row in csv_reader:
                if row[0] not in ass[keyword].keys():
                    ass[keyword][row[0]]=set()
                ass[keyword][row[0]].add(row[1])
ors=set.intersection(*[set([key for key in ass[keyword].keys() if len(ass[keyword][key])>=6]) for keyword in matrices.keys()])
dests=set.intersection(*[set.intersection(*[ass[keyword][key] for key in ass[keyword].keys() if len(ass[keyword][key])>=6]) for keyword in matrices.keys()])
cacca=""

