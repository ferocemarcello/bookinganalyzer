import time

from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

from db import db_connection, db_operator
import csv
import multiprocessing as mp

import numpy
from textblob import TextBlob
import threading
import sys

from langid.langid import LanguageIdentifier, model


punctuation_list_space=[' ',',','.',';',':','!','"','?','-','_','(',')',"'"]
identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out
def thread_function(threadnumber,sentencelist):
    toaddsents=[]
    print('len is '+str(len(sentencelist))+' for thread '+str(threadnumber))
    i=0
    for row in sentencelist:
        i+=1
        if i%100==0:
            print('i: '+str(i)+' for thread '+str(threadnumber))
        sent=row[2]
        lan = identifier.classify(sent)[0]
        acc = identifier.classify(sent)[1]
        sentan = TextBlob(sent)
        if lan == 'en' and acc >= 0.9 and sentan.polarity > 0.4 and sentan.subjectivity > 0.65:
            sent = (sentan.correct()).string
            sent = sent[0].upper() + sent[0 + 1:]
            sent = sent.replace(' i ', ' I ')

            toaddsents.append([row[0], row[1], sent])
    print(str(len(toaddsents)))

def thread_function_row_only(row):
    sent = row[2]
    lan = identifier.classify(sent)[0]
    acc = identifier.classify(sent)[1]
    sentan = TextBlob(sent)
    if lan == 'en' and acc >= 0.9 and sentan.polarity > 0.4 and sentan.subjectivity > 0.65:
        sent = (sentan.correct()).string
        sent = sent[0].upper() + sent[0 + 1:]
        sent = sent.replace(' i ', ' I ')
        return [row[0],row[1],sent]
    return

def do(originfile):
    db = db_connection()
    queryexecutor = db_operator(db)
    keywords = {}
    f = open(originfile, "r")
    for line in f:
        keyword = line[:-1]  # important to have last line with \n
        keywords[keyword] = []
        fs = open("subkeywords_booking/subkeywords_booking_cleaned/" + keyword + ".txt", "r")
        for linesub in fs:
            keywords[keyword].append(linesub[:-1])  # important to have last line with \n
        fs.close()
    f.close()
    print("Number of processors: ", mp.cpu_count())
    for emotion in ['Good', 'Bad']:
        print("begin " + emotion)
        for keyword in keywords.keys():
            print(keyword)
            subkeywords = keywords[keyword]
            toaddsents = []
            csv_file = open('csvs/' + keyword + '_' + emotion.lower() + '.csv', mode='w',encoding="utf8",newline='\n')
            csv_file.close()
            csv_file = open('csvs/all_sentences/' + keyword + '_' + emotion.lower() + '.csv', mode='r',encoding="utf8",newline='\n')
            #print("number of reviews: "+str(len(fields)))
            reader = csv.reader(csv_file, delimiter='|', quotechar='"')
            i = 0
            allsents=[]
            for row in reader:
                i+=1
                if i % 10000 == 0:
                    print(str(i))
                allsents.append(row)
                if i%100000==0:break

            csv_file.close()
            pool = mp.Pool(mp.cpu_count()*2)
            results=pool.map_async(thread_function_row_only, [row for row in allsents]).get()
            pool.close()
            pool.join()
            results=[r for r in results if r!=None]
            print("start writing sentences")
            print("num sents: " + str(len(results)))
            i = 0
            csv_file = open('csvs/' + keyword + '_' + emotion.lower() + '.csv', mode='a', encoding="utf8",newline='\n')
            for sen in results:
                i += 1
                if i % 100000 == 0:
                    print(str(i))
                writer = csv.writer(csv_file, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(sen)
            csv_file.close()
    print("done")