import csv
import os
import re
import string
import time
import multiprocessing as mp
import nltk
import spacy
import subprocess
from pycorenlp import StanfordCoreNLP
from sklearn.feature_extraction.text import TfidfVectorizer
import helper
from nltk.corpus import wordnet
import gensimldamine
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases
from nltk import word_tokenize
from spellchecker import SpellChecker
from multiprocessing import Value
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
def thread_function_row_only(row):
    text=row[2].lower()
    counter.value+=1
    if counter.value%10000==0:
        print(str(counter.value))
    for con in constr_conjs:
        if con in text:
            return None
    toks=[spell.correction(tok['lemma']) for tok in
    nlp_wrapper.annotate(text,properties={'annotators': 'lemma, pos','outputFormat': 'json',})['sentences'][0]['tokens']
    if tok['pos'] in ['NNS','NN'] and len(tok['lemma'])>1]
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
def analyze(originfile):
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
    for emotion in ['Good','Bad']:
        print("begin " + emotion)
        for keyword in list(keywords.keys()):
            if emotion=='Good' and keyword=='cleaning':
                start_time = time.time()
                print(keyword+' ---- '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
                raw_corpus = helper.getRawCorpus(
                    csv_file=open('resources/csvs/' + keyword + '_' + emotion.lower() + '.csv', mode='r',
                                  encoding="utf8", newline='\n'), additionaldetails=True)
                corpus = helper.getCorpusTextFromRaw(raw_corpus)
                spell = SpellChecker()
                counter = Value('i', 1)
                print("starting analysis")
                pool = mp.Pool(initializer=init_globals, processes=mp.cpu_count() * 2, initargs=(counter,spell,nlp_wrapper,), )
                #corpus_tok = pool.map_async(thread_function_row_only, [doc for doc in raw_corpus]).get()
                corpus_tok=[]
                for doc in corpus:
                    newdoc=False
                    doc = doc.lower()
                    counter.value += 1
                    if counter.value % 10000 == 0:
                        print(str(counter.value))
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
                        corpus_tok.append(toks)
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