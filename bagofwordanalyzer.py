import os
import string
import time

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import helper
import gensimldamine
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
def analyze(originfile):
    keywords = helper.getKeywords(originfile)
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
            for doc in corpus:
                for con in constr_conjs:
                    if con in doc.lower():
                        corpus.remove(doc)
                        break

            #take only nouns
            nlp = spacy.load("en_core_web_sm")
            #corpus=[[token for token in doc if nlp(token)[0].pos_=='NOUN']for doc in corpus[:100]]
            corpus_filt = [[str(tok).translate(str.maketrans('', '', string.punctuation)) for tok in nlp(doc) if tok.pos_ in['NOUN','PROPN'] and (str(tok).lower() not in stopwords) and len(str(tok))>1] for doc in corpus]
            ###############################################################################
            # We use the WordNet lemmatizer from NLTK. A lemmatizer is preferred over a
            # stemmer in this case because it produces more readable words. Output that is
            # easy to read is very desirable in topic modelling.
            #

            # Lemmatize the documents.
            from nltk.stem.wordnet import WordNetLemmatizer

            print("starting lemmatization")
            lemmatizer = WordNetLemmatizer()
            corpus_lemm = [[lemmatizer.lemmatize(token) for token in doc] for doc in corpus_filt]

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
            if len(corpus_lemm)>0:
                from gensim.models import Phrases
                print("doing bigrams")
                # Add bigrams and trigrams to docs (only ones that appear 10 times or more).
                bigram = Phrases(corpus_lemm, min_count=0.001*len(corpus_lemm))
                for idx in range(len(corpus_lemm)):
                    for token in bigram[corpus_lemm[idx]]:
                        if '_' in token:
                            # Token is a bigram, add to document.
                            corpus_lemm[idx].append(token)
                from gensim.corpora import Dictionary

                # Create a dictionary representation of the documents.
                dictionary = Dictionary(corpus_lemm)

                alltok = []
                freq=[]
                for doc in corpus_lemm:
                    for tok in doc:
                        alltok.append(tok)
                for t in dictionary:
                    freq.append((t,dictionary.get(t),alltok.count(dictionary.get(t)),alltok.count(dictionary.get(t))/len(alltok)))
                freq.sort(key=lambda tup: tup[2], reverse=True)
                if not os.path.exists('resources/bow/allfreq/'):
                    os.makedirs('resources/bow/allfreq/')
                with open('resources/bow/allfreq/'+keyword+'_'+emotion.lower()+'.txt', 'w') as f:
                    for item in freq:
                        f.write(str(item)+'\n')

            print('------------------------------------------------------')
            print(str(time.time() - start_time) + ' seconds to compute ' + keyword + ' ' + emotion)