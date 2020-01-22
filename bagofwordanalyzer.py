import csv
import os
import re
import string
import time
import multiprocessing as mp
import nltk
import spacy
import itertools
import subprocess
from nltk.tokenize import sent_tokenize
from pycorenlp import StanfordCoreNLP
from sklearn.feature_extraction.text import TfidfVectorizer
from langid.langid import LanguageIdentifier, model
import threading
import db
import helper
import queue
from nltk.corpus import wordnet
import gensimldamine
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases
from nltk import word_tokenize
from spellchecker import SpellChecker
from multiprocessing import Value
from multiprocessing import TimeoutError
identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
from textblob import TextBlob
negationstopset=set(['aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn','mustn', 'nan', 'negative', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn', "no", "nor", "not"])
stopset = set(
        ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
         "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
         "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
         "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
         "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
         "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up",
         "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
         "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such","only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
         "should", "now", 've', 'let', 'll','re',"etc"])
constr_conjs=set(['although','though','even if','even though','but','yet','nevertheless','however','despite','in spite'])
def init_globals(cnt,spl,nlpw):
    global counter
    global spell
    global nlp_wrapper
    counter = cnt
    spell=spl
    nlp_wrapper=nlpw
def thread_function_row_only_all(row):
    try:
        text_good=row[3].lower()
    except:
        text_good=''
    try:
        text_bad = row[4].lower()
    except:
        text_bad=''
    toks_bad=[]
    toks_good = []
    counter.value+=1
    if counter.value%10000==0:
        print(str(counter.value))
    if text_good=='':
        toks_good=[]
    else:
        for con in constr_conjs:
            if con in text_good:
                toks_good=[]
        lan = identifier.classify(text_good)[0]
        acc = identifier.classify(text_good)[1]
        if lan == 'en' and acc >= 0.9 :
            sents=sent_tokenize(text_good)
            try:

                toks_good = list(itertools.chain.from_iterable([[spell.correction(tok['lemma']) for tok in
                             nlp_wrapper.annotate(sent, properties={'annotators': 'lemma, pos', 'outputFormat': 'json', })[
                                 'sentences'][0]['tokens']
                             if tok['pos'] in ['NNS', 'NN'] and len(tok['lemma']) > 1] for sent in sents]))
            except:
                l=[]
                for sent in sents:
                    try:
                        v=[spell.correction(tok['lemma']) for tok in
                                 nlp_wrapper.annotate(sent, properties={'annotators': 'lemma, pos', 'outputFormat': 'json', })[
                                     'sentences'][0]['tokens']
                                 if tok['pos'] in ['NNS', 'NN'] and len(tok['lemma']) > 1]
                        l.append(v)
                    except Exception as e:
                        pass
                toks_good = list(itertools.chain.from_iterable(l))
            toapp = []
            for i in range(len(toks_good)):
                if '/' in toks_good[i]:
                    for tok in toks_good[i].split('/'):
                        toapp.append(tok)
            for tok in toapp:
                toks_good.append(tok)
            toapp = []
            for i in range(len(toks_good)):
                if '-' in toks_good[i]:
                    for tok in toks_good[i].split('-'):
                        toapp.append(tok)
            for tok in toapp:
                toks_good.append(tok)
    if text_bad=='':
        toks_bad=[]
    else:
        for con in constr_conjs:
            if con in text_bad:
                toks_bad=[]
        # toks=[tok for tok in toks if len(wordnet.synsets(tok)) > 0 and wordnet.synsets(tok)[0].pos() == 'n']
        lan = identifier.classify(text_bad)[0]
        acc = identifier.classify(text_bad)[1]
        if lan == 'en' and acc >= 0.9:
            sents = sent_tokenize(text_bad)
            try:

                toks_bad = list(itertools.chain.from_iterable([[spell.correction(tok['lemma']) for tok in
                             nlp_wrapper.annotate(sent, properties={'annotators': 'lemma, pos', 'outputFormat': 'json', })[
                                 'sentences'][0]['tokens']
                             if tok['pos'] in ['NNS', 'NN'] and len(tok['lemma']) > 1] for sent in sents]))
            except:
                l=[]
                for sent in sents:
                    try:
                        v=[spell.correction(tok['lemma']) for tok in
                                 nlp_wrapper.annotate(sent, properties={'annotators': 'lemma, pos', 'outputFormat': 'json', })[
                                     'sentences'][0]['tokens']
                                 if tok['pos'] in ['NNS', 'NN'] and len(tok['lemma']) > 1]
                        l.append(v)
                    except Exception as e:
                        pass
                toks_bad = list(itertools.chain.from_iterable(l))
            toapp = []
            for i in range(len(toks_bad)):
                if '/' in toks_bad[i]:
                    for tok in toks_bad[i].split('/'):
                        toapp.append(tok)
            for tok in toapp:
                toks_bad.append(tok)
            toapp = []
            for i in range(len(toks_bad)):
                if '-' in toks_bad[i]:
                    for tok in toks_bad[i].split('-'):
                        toapp.append(tok)
            for tok in toapp:
                toks_bad.append(tok)
    if toks_good+toks_bad==[]:
        return None
    return (row, toks_good + toks_bad)
def thread_function_row_only(row):
    text=row[2].lower()
    counter.value+=1
    if counter.value%100==0:
        print(str(counter.value))
    for con in constr_conjs:
        if con in text:
            return None
    try:

        toks=[spell.correction(tok['lemma']) for tok in
        nlp_wrapper.annotate(text,properties={'annotators': 'lemma, pos','outputFormat': 'json',})['sentences'][0]['tokens']
        if tok['pos'] in ['NNS','NN'] and len(tok['lemma'])>1]

    except Exception as e:
            print("fallen into Exception")
            print(str(e))
            print(str(counter.value))
            print(text)
            toks=[]
            pass
    toapp=[]
    for i in range(len(toks)):
        if '/' in toks[i]:
            for tok in toks[i].split('/'):
                toapp.append(tok)
    for tok in toapp:
        toks.append(tok)
    toapp=[]
    for i in range(len(toks)):
        if '-' in toks[i]:
            for tok in toks[i].split('-'):
                toapp.append(tok)
    for tok in toapp:
        toks.append(tok)
    #toks=[tok for tok in toks if len(wordnet.synsets(tok)) > 0 and wordnet.synsets(tok)[0].pos() == 'n']
    return (row,toks)
def analyze(originfile, all=False):
    keywords = helper.getKeywords(originfile)
    os.chdir('./resources/stanford-corenlp-full-2018-10-05')
    os.system('kill $(lsof -t -i:9000)')
    cmd = 'java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma,parse,sentiment" -port 9000 -timeout 10000000000000 &'
    time.sleep(4)
    print("starting nlp service")
    with open(os.devnull, "w") as f:
        subprocess.call(cmd, shell=True, stderr=f, stdout=f)
    time.sleep(4)
    print("nlp service started")
    os.chdir('../../')
    nlp_wrapper = StanfordCoreNLP('http://localhost:9000')
    print("Number of processors: ", mp.cpu_count())
    if all:
        print("all")
        '''if not os.path.isfile('/resources/all_test.csv'):
            print("test file created")
            open('./resources/all_test.csv', 'w').close()'''
        conn = db.db_connection()
        dbo = db.db_operator(conn)
        spell = SpellChecker()
        counter = Value('i', 1)
        corpus_tok_all=[]
        '''for i in range(1790):
            print('i=' +str(i))
            print("limit= 10000")
            print("offset= "+str(10000*i))
            conn.connect()
            query = 'SELECT reviews.ReviewID, reviews.Country as \'Tourist_Country\', ' \
                    'hotels.CountryID as \'Hotel Country\', Good, reviews.Bad ' \
                    'FROM masterthesis.reviews, masterthesis.hotels ' \
                    'where hotels.HotelNumber=reviews.HotelNumber limit 10000 offset '+str(10000*i)+';'
            results = [list(x) for x in dbo.execute(query)];
            conn.disconnect()
            print("got results from sql")
            print("starting analysis")
            print("tot number rows= " + str(len(results)))
            try:
                print('analyzing 10000 rows '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
                pool = mp.Pool(initializer=init_globals, processes=mp.cpu_count() * 2,initargs=(counter, spell, nlp_wrapper,), )
                corpus_tok = pool.map_async(thread_function_row_only_all, [doc for doc in results]).get(timeout=1200)
                pool.close()
                pool.terminate()
                pool.join()
                print('got corpus_tok for 10000 rows '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
            except TimeoutError:
                print("timeout error")
                pool.close()
                pool.terminate()
                pool.join()
                corpus_tok=[]
                for doc in results:
                    try:
                        pool = mp.Pool(initializer=init_globals, processes=mp.cpu_count() * 2, initargs=(counter,spell,nlp_wrapper,), )
                        c=pool.map_async(thread_function_row_only_all, [doc]).get(timeout=60)
                        #print('pool close')
                        pool.close()
                        pool.terminate()
                        #print('pool join')
                        pool.join()

                    except TimeoutError:
                        print(str(doc)+" caused Exception")
                        pool.close()
                        pool.terminate()
                        #print('pool join')
                        pool.join()
                        c=[None]
                    corpus_tok.append(c[0])
            print("beginning removal of sents with contrast")
            corpus_tok = [r for r in corpus_tok if r != None]
            print('len corpus_tok_reduced= '+str(len(corpus_tok)))
            corpus_tok_all+=corpus_tok
            print('len corpus_tok_all= ' + str(len(corpus_tok_all)))
            if i%100==0 and i!=0:
                with open('./resources/all_test.csv', mode='a') as file:
                    writer = csv.writer(file, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    for c in corpus_tok_all:
                        writer.writerow(c)
                file.close()
                corpus_tok_all=[]
        '''
        corpus_tok_all=[]
        i=0
        kk=set()
        with open('./resources/all_test.csv', mode='r') as file:
            reader = csv.reader(file, delimiter='|', quotechar='"')
            for row in reader:
                i+=1
                if i%10000==0:
                    print(i)
                if i==1000000:break
                k=int((row[0].split(','))[0][1:])
                if k not in kk:
                    kk.add(k)
                    corpus_tok_all.append(row)
        file.close()
        corpus_tok=corpus_tok_all
        corpustokonly = [r[1] for r in corpus_tok] 
        print("doing bigrams")
        # Add bigrams and trigrams to docs (only ones that appear 10 times or more).
        bigram = Phrases(corpustokonly, min_count=0.001 * len(corpus_tok))
        lenc=len(corpus_tok)
        print(lenc)
        print(corpus_tok[0][1])
        for idx in range(lenc):
            if idx%10000==0:
                print(idx)
            for token in bigram[corpustokonly[idx]]:
                if '_' in token:
                    # Token is a bigram, add to document.
                    try:
                        corpus_tok[idx][1].append(token)
                    except Exception as e:
                        pass
        from gensim.corpora import Dictionary
        print("writing frequence file")
        '''all_set=set()
        for emotion in ['Good', 'Bad']:
            print("begin " + emotion)
            for keyword in list(keywords.keys()):
                if not (keyword == 'cleaning' or keyword=='pet'):
                    start_time = time.time()
                    print(keyword + ' ---- ' + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
                    raw_corpus = helper.getRawCorpus(
                        csv_file=open('resources/csvs/' + keyword + '_' + emotion.lower() + '.csv', mode='r',
                                      encoding="utf8", newline='\n'), additionaldetails=True)
                    # corpus = helper.getCorpusTextFromRaw(raw_corpus)
                    spell = SpellChecker()
                    counter = Value('i', 1)
                    print("starting analysis")
                    pool = mp.Pool(initializer=init_globals, processes=mp.cpu_count() * 2,
                                   initargs=(counter, spell, nlp_wrapper,), )
                    corpus_tok = pool.map_async(thread_function_row_only, [doc for doc in raw_corpus]).get()
                    print('pool close')
                    pool.close()
                    print('pool join')
                    pool.join()
                    print("beginning removal of sents with contrast")
                    corpus_tok = [r for r in corpus_tok if r != None]
                    ###############################################################################
                    # We find bigrams in the documents. Bigrams are sets of two adjacent words.
                    # Using bigrams we can get phrases like "machine_learning" in our output
                    # (spaces are replaced with underscores); without bigrams we would only get
                    # "machine" and "learning".
                    #
                    # Note that in the code below, we find bigrams and then add them to the
                    # original data, because we would like to keep the words "machine" and
                    # "learning" as well as the bigram "machine_learning".
                    #
                    # .. Important::
                    #     Computing n-grams of large dataset can be very computationally
                    #     and memory intensive.
                    #
                    print('len all_set_tok before= ' + str(len(all_set)))
                    print('len corpus_tok= ' + str(len(corpus_tok)))
                    print('len corpus_tok+all_set_tok= ' + str(len(corpus_tok) + len(all_set)))
                    for sen in corpus_tok:
                        all_set.add((tuple(sen[0]),tuple(sen[1])))
                    print('len all_set_tok after= ' + str(len(all_set)))
                    print('------------------------------------------------------')
                    print(str(time.time() - start_time) + ' seconds to compute ' + keyword + ' ' + emotion)
        # Compute bigrams.
        if len(all_set) > 0:
            corpus_tok=[(list(x[0]),list(x[1])) for x in all_set]
            corpustokonly = [r[1] for r in corpus_tok]
            print("doing bigrams")
            # Add bigrams and trigrams to docs (only ones that appear 10 times or more).
            bigram = Phrases(corpustokonly, min_count=0.001 * len(corpus_tok))
            for idx in range(len(corpus_tok)):
                for token in bigram[corpustokonly[idx]]:
                    if '_' in token:
                        # Token is a bigram, add to document.
                        corpus_tok[idx][1].append(token)
            from gensim.corpora import Dictionary
            print("writing frequence file")

            # Create a dictionary representation of the documents.
            dictionary = Dictionary(corpustokonly)

            alltok = []
            freq = []
            for doc in corpustokonly:
                for tok in doc:
                    alltok.append(tok)
            lencorpus = len(corpus_tok)
            print("len dictionary = " + str(len(dictionary.keys())))
            i = 0
            for t in dictionary:
                i += 1
                if i % 1000 == 0:
                    print("analyzing token " + str(i))
                freqsent = 0
                for doc in corpustokonly:
                    if dictionary.get(t) in doc:
                        freqsent += 1
                freq.append((t, dictionary.get(t), alltok.count(dictionary.get(t)),
                             alltok.count(dictionary.get(t)) / len(alltok), freqsent, freqsent / lencorpus))
            freq.sort(key=lambda tup: tup[5], reverse=True)
            for i in range(len(freq)):
                freq[i] = tuple(list(freq[i]) + [i])
            if not os.path.exists('resources/bow/allfreq/stanford/'):
                os.makedirs('resources/bow/allfreq/stanford/')
            with open('resources/bow/allfreq/stanford/all.txt',
                      'w') as f:
                for item in freq:
                    f.write(str(item) + '\n')
                f.close()

            print("writing bow file")
            top_tokens = [f[1] for f in freq[:500]]
            lentoptok = len(top_tokens)
            corpus_bow = {}
            toplen = 0
            for i in range(len(corpus_tok)):
                corpus_bow[i] = [0] * lentoptok
                if len(corpus_tok[i][0] + corpus_tok[i][1]) > toplen:
                    toplen = len(corpus_tok[i][0] + corpus_tok[i][1])
                for tok in corpus_tok[i][1]:
                    if tok in top_tokens:
                        corpus_bow[i][top_tokens.index(tok)] = 1

            with open('resources/bow/all.csv', mode='w') as file:
                writer = csv.writer(file, delimiter='|', quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)
                writer.writerow([''] * toplen + top_tokens)
                for i in corpus_bow.keys():
                    writer.writerow(corpus_tok[i][0] + corpus_tok[i][1] + [''] * (
                            toplen - len(corpus_tok[i][0] + corpus_tok[i][1])) + corpus_bow[i])
            file.close()
        '''
        # Create a dictionary representation of the documents.
        dictionary = Dictionary(corpustokonly)

        alltok = []
        freq = []
        for doc in corpustokonly:
            for tok in doc:
                alltok.append(tok)
        lencorpus = len(corpus_tok)
        print("len dictionary = " + str(len(dictionary.keys())))
        i = 0
        for t in dictionary:
            i += 1
            if i % 1000 == 0:
                print("analyzing token " + str(i))
            freqsent = 0
            for doc in corpustokonly:
                if dictionary.get(t) in doc:
                    freqsent += 1
            freq.append((t, dictionary.get(t), alltok.count(dictionary.get(t)),
                         alltok.count(dictionary.get(t)) / len(alltok), freqsent, freqsent / lencorpus))
        freq.sort(key=lambda tup: tup[5], reverse=True)
        for i in range(len(freq)):
            freq[i] = tuple(list(freq[i]) + [i])
        if not os.path.exists('resources/bow/allfreq/stanford/'):
            os.makedirs('resources/bow/allfreq/stanford/')
        with open('resources/bow/allfreq/stanford/all.txt', 'w') as f:
            for item in freq:
                f.write(str(item) + '\n')
            f.close()

        print("writing bow file")
        top_tokens = [f[1] for f in freq[:500]]
        lentoptok = len(top_tokens)
        corpus_bow = {}
        toplen = 0
        for i in range(len(corpus_tok)):
            corpus_bow[i] = [0] * lentoptok
            if len(corpus_tok[i][0] + corpus_tok[i][1]) > toplen:
                toplen = len(corpus_tok[i][0] + corpus_tok[i][1])
            for tok in corpus_tok[i][1]:
                if tok in top_tokens:
                    corpus_bow[i][top_tokens.index(tok)] = 1

        with open('resources/bow/all.csv', mode='w') as file:
            writer = csv.writer(file, delimiter='|', quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)
            writer.writerow([''] * toplen + top_tokens)
            for i in corpus_bow.keys():
                writer.writerow(
                    corpus_tok[i][0] + corpus_tok[i][1] + [''] * (toplen - len(corpus_tok[i][0] + corpus_tok[i][1])) +
                    corpus_bow[i])
        file.close()
    else:
        print("not all")
        for emotion in ['Good','Bad']:
            print("begin " + emotion)
            for keyword in list(keywords.keys()):
                if emotion=='Good' and keyword=='cleaning':#cleaning good
                    start_time = time.time()
                    print(keyword+' ---- '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
                    spell = SpellChecker()
                    counter = Value('i', 1)
                    corpus_tok_all=[]
                    #if not os.path.isfile('/resources/cleaning_test.csv'):
                        #open('./resources/cleaning_test.csv', 'w').close()
                    for i in range(400):#400
                        print(str(i))
                        offset=i*1000
                        limit=1000
                        print("starting reading")
                        print("limit="+str(limit))
                        print("offset="+str(offset))
                        raw_corpus = helper.getRawCorpus(
                            csv_file=open('resources/csvs/' + keyword + '_' + emotion.lower() + '.csv', mode='r',
                                          encoding="utf8", newline='\n'), additionaldetails=True, limit=limit, offset=offset)

                        #corpus = helper.getCorpusTextFromRaw(raw_corpus)
                        #raw_corpus_half_one = raw_corpus[:int(len(raw_corpus) / 2)]
                        #raw_corpus_half_two=raw_corpus[int(len(raw_corpus)/2):]
                        print("starting analysis")
                        pool = mp.Pool(initializer=init_globals, processes=mp.cpu_count() * 2, initargs=(counter,spell,nlp_wrapper,), )
                        try:
                            corpus_tok = pool.map_async(thread_function_row_only, [doc for doc in raw_corpus]).get(timeout=30)
                            pool.close()
                            pool.join()
                        except TimeoutError:
                            print("timeout error")
                            print('pool close')
                            pool.close()
                            print('pool terminate')
                            pool.terminate()
                            print('pool join')
                            pool.join()
                            corpus_tok=[]
                            for doc in raw_corpus:
                                try:
                                    pool = mp.Pool(initializer=init_globals, processes=mp.cpu_count() * 2, initargs=(counter,spell,nlp_wrapper,), )
                                    c=pool.map_async(thread_function_row_only, [doc]).get(timeout=30)
                                    #print('pool close')
                                    pool.close()
                                    #print('pool join')
                                    pool.join()
                                    '''thread = threading.Thread(target = thread_function_row_only, args = (doc))
                                    thread.start()
                                    thread.join()
                                    c=que.get()'''
                                except TimeoutError:
                                    print(str(doc)+" caused Exception")
                                    c=[None]
                                corpus_tok.append(c[0])
                        corpus_tok_reduced=[r for r in corpus_tok if r != None]
                        print("len corpus_tok: " + str(len(corpus_tok)))
                        print("len corpus_tok_reduced: " + str(len(corpus_tok_reduced)))
                        '''with open('./resources/cleaning_test.csv', mode='a') as file:
                            writer = csv.writer(file, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            for c in corpus_tok_reduced:
                                writer.writerow(c)
                        file.close()'''
                        corpus_tok_all+=corpus_tok_reduced
                        print("len corpus_tok_all: " + str(len(corpus_tok_all)))
                    '''
                    corpus_tok=[]
                    s=0
                    for doc in corpus:
                        newdoc=False
                        doc = doc.lower()
                        s += 1
                        if s % 10000 == 0:
                            print(str(s))
                        for con in constr_conjs:
                            if con in doc:
                                newdoc=True
                                break
                        if not newdoc:
                            toks = [spell.correction(tok['lemma']) for tok in
                                    nlp_wrapper.annotate(doc,
                                                         properties={'annotators': 'lemma, pos', 'outputFormat': 'json', })[
                                        'sentences'][0]['tokens']
                                    if tok['pos'] in ['NNS', 'NN'] and len(tok['lemma']) > 1]
                            toapp = []
                            for i in range(len(toks)):
                                if '/' in toks[i]:
                                    for tok in toks[i].split('/'):
                                        toapp.append(tok)
                            for tok in toapp:
                                toks.append(tok)
                            toapp = []
                            for i in range(len(toks)):
                                if '-' in toks[i]:
                                    for tok in toks[i].split('-'):
                                        toapp.append(tok)
                            for tok in toapp:
                                toks.append(tok)
                            corpus_tok.append(toks)'''
                    #print("beginning removal of sents with contrast")
                    corpus_tok=corpus_tok_all
                    print("len corpus_tok: " + str(len(corpus_tok)))
                    ###############################################################################
                    # We find bigrams in the documents. Bigrams are sets of two adjacent words.
                    # Using bigrams we can get phrases like "machine_learning" in our output
                    # (spaces are replaced with underscores); without bigrams we would only get
                    # "machine" and "learning".
                    #
                    # Note that in the code below, we find bigrams and then add them to the
                    # original data, because we would like to keep the words "machine" and
                    # "learning" as well as the bigram "machine_learning".
                    #
                    # .. Important::
                    #     Computing n-grams of large dataset can be very computationally
                    #     and memory intensive.
                    #
                    # Compute bigrams.
                    if len(corpus_tok)>0:
                        corpustokonly=[r[1] for r in corpus_tok]
                        print("doing bigrams")
                        # Add bigrams and trigrams to docs (only ones that appear 10 times or more).
                        bigram = Phrases(corpustokonly, min_count=0.001 * len(corpus_tok))
                        for idx in range(len(corpus_tok)):
                            for token in bigram[corpustokonly[idx]]:
                                if '_' in token:
                                    # Token is a bigram, add to document.
                                    corpus_tok[idx][1].append(token)
                        from gensim.corpora import Dictionary
                        print("writing frequence file")

                        # Create a dictionary representation of the documents.
                        dictionary = Dictionary(corpustokonly)

                        alltok = []
                        freq=[]
                        for doc in corpustokonly:
                            for tok in doc:
                                alltok.append(tok)
                        lencorpus=len(corpus_tok)
                        print("len dictionary = "+str(len(dictionary.keys())))
                        i=0
                        for t in dictionary:
                            i+=1
                            if i%1000==0:
                                print("analyzing token "+str(i))
                            freqsent = 0
                            for doc in corpustokonly:
                                if dictionary.get(t) in doc:
                                    freqsent+=1
                            freq.append((t,dictionary.get(t),alltok.count(dictionary.get(t)),alltok.count(dictionary.get(t))/len(alltok),freqsent,freqsent/lencorpus))
                        freq.sort(key=lambda tup: tup[5], reverse=True)
                        for i in range(len(freq)):
                            freq[i]=tuple(list(freq[i])+[i])
                        if not os.path.exists('resources/bow/allfreq/stanford/'):
                            os.makedirs('resources/bow/allfreq/stanford/')
                        with open('resources/bow/allfreq/stanford/'+keyword+'_'+emotion.lower()+'.txt', 'w') as f:
                            for item in freq:
                                f.write(str(item)+'\n')
                            f.close()

                        print("writing bow file")
                        top_tokens=[f[1] for f in freq[:500]]
                        lentoptok=len(top_tokens)
                        corpus_bow={}
                        toplen=0
                        for i in range(len(corpus_tok)):
                            corpus_bow[i]=[0]*lentoptok
                            if len(corpus_tok[i][0]+corpus_tok[i][1])>toplen:
                                toplen=len(corpus_tok[i][0]+corpus_tok[i][1])
                            for tok in corpus_tok[i][1]:
                                if tok in top_tokens:
                                    corpus_bow[i][top_tokens.index(tok)]=1

                        with open('resources/bow/'+keyword+'_'+emotion.lower()+'.csv', mode='w') as file:
                            writer = csv.writer(file, delimiter='|', quotechar='"',
                                                         quoting=csv.QUOTE_MINIMAL)
                            writer.writerow(['']*toplen+top_tokens)
                            for i in corpus_bow.keys():
                                writer.writerow(corpus_tok[i][0]+corpus_tok[i][1]+['']*(toplen-len(corpus_tok[i][0]+corpus_tok[i][1]))+corpus_bow[i])
                        file.close()
                    print('------------------------------------------------------')
                    print(str(time.time() - start_time) + ' seconds to compute ' + keyword + ' ' + emotion)
    f.close()
