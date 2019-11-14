import os
import re
import string
import time
import multiprocessing as mp
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import helper
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
def init_globals(cnt,spl):
    global counter
    global spell
    counter = cnt
    spell=spl
def thread_function_row_only(row):
    row= row.lower()
    counter.value+=1
    for con in constr_conjs:
        if con in row:
            return ""
    toks = word_tokenize(row)
    cors = []
    for tok in toks:
        sub = re.sub('[^A-Za-z]+', ' ', tok)
        cor = spell.correction(sub).split(' ')
        for i in range(len(cor)):
            if not cor[i].isalpha():
                cor[i] = tok
            cors.append(cor[i])
    row= ' '.join(cors)
    return row
def analyze(originfile):
    keywords = helper.getKeywords(originfile)
    print("Number of processors: ", mp.cpu_count())
    for emotion in ['Good','Bad']:
        print("begin " + emotion)
        for keyword in list(keywords.keys()):
            start_time = time.time()
            print(keyword)
            raw_corpus = helper.getRawCorpus(
                csv_file=open('resources/csvs/' + keyword + '_' + emotion.lower() + '.csv', mode='r',
                              encoding="utf8", newline='\n'), id_and_country=True,additionaldetails=True)
            corpus = helper.getCorpusTextFromRaw(raw_corpus)
            stopwords = gensimldamine.getStopwords(stopset)
            stwfromtfidf = list(TfidfVectorizer(stop_words='english').get_stop_words())
            stopwords = set(list(stopwords) + stwfromtfidf)
            for w in negationstopset:
                stopwords.add(w)
            nlp = spacy.load("en_core_web_sm")
            lemmatizer = WordNetLemmatizer()
            spell = SpellChecker()
            counter = Value('i', 0)
            pool = mp.Pool(initializer=init_globals, processes=mp.cpu_count() * 2, initargs=(counter,spell,), )
            corpus = pool.map_async(thread_function_row_only, [doc for doc in corpus]).get()
            pool.close()
            pool.join()
            corpus = [r for r in corpus if r != ""]
            #take only nouns
            #corpus=[[token for token in doc if nlp(token)[0].pos_=='NOUN']for doc in corpus[:100]]
            print("doing pos tagging")
            corpus_filt_spacy = [#re.sub('[^A-Za-z0-9]+', '', str(tok))
                [lemmatizer.lemmatize(str(tok)) for tok in nlp(doc) if
                 tok.pos_ == 'NOUN' and (str(tok) not in stopwords)] for doc in corpus]
            for i in range(len(corpus_filt_spacy)):
                corpus_filt_spacy[i]=list(filter(lambda x: len(x) > 1 and x.isalpha(), corpus_filt_spacy[i]))
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
            if len(corpus_filt_spacy)>0:
                print("doing bigrams")
                # Add bigrams and trigrams to docs (only ones that appear 10 times or more).
                bigram = Phrases(corpus_filt_spacy, min_count=0.001*len(corpus_filt_spacy))
                for idx in range(len(corpus_filt_spacy)):
                    for token in bigram[corpus_filt_spacy[idx]]:
                        if '_' in token:
                            # Token is a bigram, add to document.
                            corpus_filt_spacy[idx].append(token)
                from gensim.corpora import Dictionary

                # Create a dictionary representation of the documents.
                dictionary = Dictionary(corpus_filt_spacy)

                alltok = []
                freq=[]
                for doc in corpus_filt_spacy:
                    for tok in doc:
                        alltok.append(tok)
                for t in dictionary:
                    freq.append((t,dictionary.get(t),alltok.count(dictionary.get(t)),alltok.count(dictionary.get(t))/len(alltok)))
                freq.sort(key=lambda tup: tup[2], reverse=True)
                for i in range(len(freq)):
                    freq[i]=tuple(list(freq[i])+[i])
                if not os.path.exists('resources/bow/allfreq/'):
                    os.makedirs('resources/bow/allfreq/')
                with open('resources/bow/allfreq/'+keyword+'_'+emotion.lower()+'.txt', 'w') as f:
                    for item in freq:
                        f.write(str(item)+'\n')
                    f.close()

            print('------------------------------------------------------')
            print(str(time.time() - start_time) + ' seconds to compute ' + keyword + ' ' + emotion)