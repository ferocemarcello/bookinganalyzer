import time

from nltk import word_tokenize

import helper
from db import db_connection, db_operator
import csv
import multiprocessing as mp
from multiprocessing import Value
'''import spacy
from spacy_langdetect import LanguageDetector
nlp = spacy.load("en")'''
from textblob import TextBlob
import threading
import sys
'''import grammar_check
from gingerit.gingerit import GingerIt'''
from langid.langid import LanguageIdentifier, model
'''from spacy.tokens import Doc, Span
from googletrans import Translator'''

'''def custom_detection_function(spacy_object):
    # custom detection function should take a Spacy Doc or a
    assert isinstance(spacy_object, Doc) or isinstance(
        spacy_object, Span), "spacy_object must be a spacy Doc or Span object but it is a {}".format(type(spacy_object))
    detection = Translator().detect(spacy_object.text)
    return {'language':detection.lang, 'score':detection.confidence}'''

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
        if i%100000==0:
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
    with counter.get_lock():
        counter.value += 1
    if counter.value%100000==0:
        print('review '+str(counter.value))

    words = word_tokenize(sent)
    skipsentence = True
    for word in words:
        if word.lower() in subks._getvalue():
            skipsentence = False
            break
    if not skipsentence:
        if lan == 'en' and acc >= 0.9 and sentan.polarity > 0.4 and sentan.subjectivity > 0.65:
            return [row[0], row[1], sent]
    #if lan == 'en' and acc >= 0.9 and sentan.polarity > 0.4 and sentan.subjectivity > 0.65:
    # sent = (sentan.correct()).string
        '''sent = sent[0].upper() + sent[0 + 1:]
        sent = sent.replace(' i ', ' I ')'''
    return

def init_globals(cnt,subkeywords):
    global counter
    counter = cnt
    global subks
    subks=subkeywords

def do(originfile):
    start_time = time.time()
    db = db_connection()
    queryexecutor = db_operator(db)
    keywords=helper.getKeywords(originfile)
    #nlp.add_pipe(LanguageDetector(), name="language_detector", last=True)
    #nlp.add_pipe(LanguageDetector(language_detection_function=custom_detection_function), name="language_detector",last=True)
    print("Number of processors: ", mp.cpu_count())
    for emotion in ['Good', 'Bad']:
        print("begin " + emotion)
        for keyword in keywords.keys():
            print(keyword)
            '''f=open('resources/csvs/' + keyword + '_' + emotion.lower() + '.csv', mode='w')
            f.close()
            liketext = 'SELECT ReviewID, Country, ' + emotion + ' from masterthesis.reviews where '
            '''
            subkeywords = keywords[keyword]
            '''for subkey in subkeywords:
                liketext += emotion + " LIKE '%" + subkey + "%' or "
            liketext = liketext[:-4]
            #liketext+=" limit 10000;"
            liketext += ";"
            db.connect()
            fields = queryexecutor.execute(query=liketext)
            db.disconnect()
            print("start analyzing sentences")'''
            toaddsents = []
            csv_file = open('resources/csvs/' + keyword + '_' + emotion.lower() + '.csv', mode='w',encoding="utf8",newline='\n')
            csv_file.close()
            csv_file = open('resources/csvs/all_sentences/' + keyword + '_' + emotion.lower() + '.csv', mode='r',encoding="utf8",newline='\n')
            '''row_count = sum(1 for row in reader)
            print("number of sentences: " + str(len(row_count)))'''
            #print("number of reviews: "+str(len(fields)))
            reader = csv.reader(csv_file, delimiter='|', quotechar='"')
            i = 0
            allsents=[]
            for row in reader:
                i+=1
                if i % 100000 == 0:
                    print('reading sentence '+str(i))
                allsents.append(row)
                #if i%1000==0:break
                #sent=row[2]
                '''lan = identifier.classify(sent)[0]
                acc = identifier.classify(sent)[1]
                if lan == 'en' and acc >= 0.9:
                    sentan = TextBlob(sent)
                    sent = (sentan.correct()).string
                    sentan = TextBlob(sent)
                    words = word_tokenize(sent)
                    skipsentence = True
                    for word in words:
                        if word.lower() in subkeywords:
                            skipsentence = False
                            break
                    if not skipsentence:
                        # tool = grammar_check.LanguageTool('en-GB')
                        # matches = tool.check(text)
                        sent = sent[0].upper() + sent[0 + 1:]
                        sent = sent.replace(' i ', ' I ')
                        #toaddsents.append([row[0],row[1],sent])
                        # sentan.sentiment_assessments
                        if sentan.polarity > 0.4 and sentan.subjectivity > 0.65:
                            toaddsents.append([row[0],row[1],sent])'''
            csv_file.close()
            print('num of reviews: '+str(len(allsents)))
            counter = Value('i', 0)
            subks=mp.Manager().list(subkeywords)
            pool = mp.Pool(initializer=init_globals, processes=mp.cpu_count()*2,initargs=(counter,subks,),)
            results=pool.map_async(thread_function_row_only, [row for row in allsents]).get()
            pool.close()
            pool.join()
            results=[r for r in results if r!=None]
            '''l = chunkIt(allsents,2)
            ths = []
            threads = list()
            for index in range(2):
                li=l[index]
                x = threading.Thread(target=thread_function, args=(index,li,))
                threads.append(x)
                x.start()
            for index,thread in enumerate(threads):
                thread.join()'''
            '''for rew in fields:
                i+=1
                if i%10000==0:
                    print(str(i))
                text = rew[2]
                text = text.replace('\n', ' ')
                text = text.replace('\t', ' ')
                text = text.replace('\r', ' ')
                text = text.replace('.', '. ')
                text = text.replace(':', ': ')
                text = text.replace(';', '; ')
                text = text.replace(',', ', ')
                lan=identifier.classify(text)[0]
                acc=identifier.classify(text)[1]
                if lan == 'en' and acc >= 0.9:
                    tokenized_text = sent_tokenize(text)
                    for sent in tokenized_text:
                        sentan = TextBlob(sent)
                        sent = (sentan.correct()).string
                        sentan=TextBlob(sent)
                        words = word_tokenize(sent)
                        skipsentence = True
                        for word in words:
                            if word.lower() in subkeywords:
                                skipsentence = False
                                break
                        if not skipsentence:
                            # tool = grammar_check.LanguageTool('en-GB')
                            # matches = tool.check(text)
                            sent = sent[0].upper() + sent[0 + 1:]
                            sent = sent.replace(' i ', ' I ')
                            # sentan.ngrams(3)
                            # sentan.sentiment_assessments
                            if sentan.polarity > 0.4 and sentan.subjectivity > 0.65:
                                toaddsents.append([rew[0], rew[1], sent])
                tokenized_text = sent_tokenize(text)
                for sent in tokenized_text:
                    toaddsents.append([rew[0], rew[1], sent])'''
            print("start writing sentences")
            print("num sents: " + str(len(results)))
            i = 0
            csv_file = open('resources/csvs/' + keyword + '_' + emotion.lower() + '.csv', mode='a', encoding="utf8",newline='\n')
            for sen in results:
                i += 1
                if i % 100000 == 0:
                    print('writing sent '+str(i))
                writer = csv.writer(csv_file, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(sen)
            csv_file.close()
    print("done")
    print("--- %s seconds ---" % (time.time() - start_time))