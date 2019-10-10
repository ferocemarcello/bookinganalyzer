import csv

import helper
import fasttext
from collections import defaultdict
from gensim import corpora
from gensim import models

class TopicWriter:

    def do(self,originfile):
        keywords=helper.getKeywords(originfile)
        #https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/gensim%20Quick%20Start.ipynb
        # Create a set of frequent words
        #https://gist.github.com/sebleier/554280
        stopset=set(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"])
        punctuation_list = ['+',',', '.', ';', ':', '!', '"', '?', '-', '_', '(', ')', "'"]
        for emotion in ['Good', 'Bad']:
            print("begin " + emotion)
            for keyword in keywords.keys():
                raw_corpus = []
                print(keyword)
                csv_file = open('resources/csvs/' + keyword + '_' + emotion.lower() + '.csv', mode='r',
                                encoding="utf8", newline='\n')
                reader = csv.reader(csv_file, delimiter='|', quotechar='"')
                subkeywords = keywords[keyword]
                i=0
                for row in reader:
                    i+=1
                    if i % 50000 == 0:
                        print('reading sentence '+str(i))
                    raw_corpus.append(row[2])
                '''vectorrep,dictionary=self.getVectorRepresentation(raw_corpus,stopset)
                tfidfmod=self.getTfIdfModel(vectorrep,dictionary)'''
                #https://towardsdatascience.com/the-complete-guide-for-topics-extraction-in-python-a6aaa6cedbbc
                list_of_list_of_tokens = [[word for word in document.lower().split() if word not in stopset]
                         for document in raw_corpus[:100000]]
                ind=0
                newlist=[]
                for l in list_of_list_of_tokens:
                    for w in l:
                        for p in punctuation_list:
                            if p in w:
                                nw = w.replace(p, '')
                                l[l.index(w)] = nw
                                w=nw
                                break
                        l[l.index(w)]=w.replace(' ','')
                    if len(l)!=0:
                        newlist.append(l)

                list_of_list_of_tokens=newlist
                dictionary_LDA = corpora.Dictionary(list_of_list_of_tokens)
                dictionary_LDA.filter_extremes(no_below=0.01*len(list_of_list_of_tokens))
                corpus = [dictionary_LDA.doc2bow(list_of_tokens) for list_of_tokens in list_of_list_of_tokens]
                #lda_model = models.LdaModel(corpus, id2word=dictionary_LDA)
                num_topics = 20
                lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary_LDA, passes=4, alpha=[0.01] * num_topics, eta=[0.01] * len(dictionary_LDA.keys()))
                tops=(lda_model.show_topics(num_topics=num_topics, num_words=100, log=False, formatted=True))
                ratiotopiccorpuszero=lda_model[corpus[0]]  # corpus[0] means the first document.

                print(lda_model[dictionary_LDA.doc2bow(['milk','extremely','fresh'])])

    def getVectorRepresentation(self, raw_corpus,stopset):
    # Lowercase each document, split it by white space and filter out stopwords
        texts = [[word for word in document.lower().split() if word not in stopset]
             for document in raw_corpus]

        # Count word frequencies
        frequency = defaultdict(int)
        for text in texts:
            for token in text:
                frequency[token] += 1

        # Only keep words that appear more than once
        processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
        dictionary = corpora.Dictionary(processed_corpus)
        bow_corpus = []
        print('len corpus: ' + str(len(processed_corpus)))
        i = 0
        for text in processed_corpus:
            i += 1
            bow_corpus.append(dictionary.doc2bow(text))
        # bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
        return bow_corpus,dictionary
    def getTfIdfModel(self,bow_corpus,dictionary):
        # The first entry in each tuple corresponds to the ID of the token in the dictionary, the second corresponds to the count of this token.
        # train the model
        tfidf = models.TfidfModel(bow_corpus)
        # transform the "system minors" string
        breakfastcoffee = tfidf[dictionary.doc2bow("breakfast coffee".lower().split())]
        # The tfidf model again returns a list of tuples, where the first entry is the token ID and the second entry is the tf-idf weighting.