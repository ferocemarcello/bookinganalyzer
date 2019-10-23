import copy
import csv
import math

import matplotlib
from scipy.stats import norm
import helper
from collections import defaultdict
import gensim
from gensim import corpora
from gensim import models
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import re, nltk, spacy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import seaborn as sns;sns.set()  # for plot styling
import numpy as np
from sklearn.metrics import silhouette_score
import os

class TopicWriter:
    # Create a set of frequent words
    # https://gist.github.com/sebleier/554280
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
    negationstopset=set(['aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn','mustn', 'nan', 'negative', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn', "no", "nor", "not"])
    punctuation_list = ['+', ',', '.', ';', ':', '!', '"', '?', '-', '_', '(', ')', "'"]
    def getTokensCleanStopset(self, raw_corpus, stopset,limit):
        list_of_list_of_tokens = [[word for word in document.lower().split() if word not in stopset]
                                  for document in raw_corpus[:limit]]
        return list_of_list_of_tokens
    def getVectorRepresentation(self, tokens):
        # Count word frequencies
        frequency = defaultdict(int)
        for text in tokens:
            for token in text:
                frequency[token] += 1

        # Only keep words that appear more than once
        processed_corpus = [[token for token in text if frequency[token] > 1] for text in tokens]
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
    def cleantokenspuncts(self, list_of_list_of_tokens,punctuation_list):
        newlist = []
        for l in list_of_list_of_tokens:
            for w in l:
                for p in punctuation_list:
                    if p in w:
                        nw = w.replace(p, '')
                        l[l.index(w)] = nw
                        w = nw
                        break
                l[l.index(w)] = w.replace(' ', '')
            if len(l) != 0:
                newlist.append(l)
        return newlist
    def getNationsCountDict(self, csv_file):
        reader = csv.reader(csv_file, delimiter='|', quotechar='"')
        nat_dict_count={}
        for row in reader:
            if row[1] not in nat_dict_count.keys():
                nat_dict_count[row[1]]=1
            else:
                nat_dict_count[row[1]]+=1
        csv_file.close()
        return nat_dict_count
    def getRawCorpus(self, csv_file,all=False):
        raw_corpus = []
        reader = csv.reader(csv_file, delimiter='|', quotechar='"')
        i = 0
        if all:
            for row in reader:
                i += 1
                if i % 50000 == 0:
                    print('reading sentence ' + str(i))
                raw_corpus.append(row)
        else:
            for row in reader:
                i += 1
                if i % 50000 == 0:
                    print('reading sentence ' + str(i))
                raw_corpus.append(row[2])
        csv_file.close()
        return raw_corpus
    def getStopwords(self, stopset):
        stopwords = set(STOPWORDS)
        stopwords.update(stopset)
        return stopwords
    def display_save_wordcloud(self, wordcloud, display=False, save=False,path=None):
        plt.figure(figsize=(10, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        if display:plt.show()
        # Save the image in the img folder:
        if save:wordcloud.to_file(path)
        plt.close(plt.figure())
    def sent_to_words(self,sentences):
        for sentence in sentences:
            yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))#deacc=True removes punctuations
    def lemmatization(self,texts, nlp,allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if
                                       token.pos_ in allowed_postags]))
        return texts_out
    def color_green(self,val):
        color = 'green' if val > .1 else 'black'
        return 'color: {col}'.format(col=color)
    def make_bold(self,val):
        weight = 700 if val > .1 else 400
        return 'font-weight: {weight}'.format(weight=weight)
    def display_topics(self,model, feature_names, no_top_words):
        for topic_idx, topic in enumerate(model.components_):
            print("Topic %d:" % (topic_idx))
            print(" ".join([feature_names[i]+'('+str(topic[i])+')'
                            for i in topic.argsort()[:-no_top_words - 1:-1]]))
    def show_lda_topics(self,lda_model=None, n_words=20,df_topic_keywords=None):
        keywords = np.array(df_topic_keywords.columns)
        topic_keywords = []
        newcomps=lda_model.components_ / lda_model.components_.sum(axis=1)[:, np.newaxis]
        for topic_weights in newcomps:
            top_keyword_locs = (-topic_weights).argsort()[:n_words]
            kw=keywords.take(top_keyword_locs)
            w=topic_weights.take(top_keyword_locs)
            kww=[str(kw[i])+'('+str(w[i])+')' for i in range(len(kw))]
            #topic_keywords.append(keywords.take(top_keyword_locs))
            topic_keywords.append(kww)
        return topic_keywords
    def create_wordcloud(self, corpus=[],stopwords=[],display=False, save=True, path=""):
        text = " ".join(review for review in corpus)
        wordcloud = WordCloud(stopwords=stopwords, background_color="white", max_words=100).generate(text)
        # Display the generated image:
        # the matplotlib way:
        self.display_save_wordcloud(wordcloud, display=display, save=save,path=path)
    def cluster_raw_corpus_by_nation(self, raw_corpus):
        raw_corpus_by_nation={}
        for r in raw_corpus:
            if r[1]!='':
                if r[1] not in raw_corpus_by_nation.keys():
                    raw_corpus_by_nation[r[1]]=[]
                raw_corpus_by_nation[r[1]].append(r)
        return raw_corpus_by_nation
    def get_raw_corpus_nat(self, nat, raw_corpus_by_nation):
        reviews=raw_corpus_by_nation[nat]
        return reviews
    def do(self,originfile):
        keywords = helper.getKeywords(originfile)
        # Create stopword list:
        stopwords = self.getStopwords(self.stopset)
        stwfromtfidf=list(TfidfVectorizer(stop_words='english').get_stop_words())
        stopwords=set(list(stopwords)+stwfromtfidf)
        negation = False
        if negation:
            for w in self.negationstopset:
                stopwords.add(w)
        else:
            for w in self.negationstopset:
                try:
                    stopwords.remove(w)
                except:
                    None#word already not in stopwords
        for emotion in ['Good','Bad']:
            print("begin " + emotion)
            for keyword in list(keywords.keys()):
                print(keyword)
                raw_corpus = self.getRawCorpus(
                    csv_file=open('resources/csvs/' + keyword + '_' + emotion.lower() + '.csv', mode='r',
                                  encoding="utf8", newline='\n'),all=True)
                raw_corpus_by_nation = self.cluster_raw_corpus_by_nation(raw_corpus)
                todeletenations=[]
                for k in raw_corpus_by_nation.keys():
                    if len(raw_corpus_by_nation[k])<100:
                        todeletenations.append(k)
                raw_corpus=[r for r in raw_corpus if r[1] not in todeletenations]
                corpus = self.getCorpusTextFromRaw(raw_corpus)
                self.doKaggle(corpus, stopwords,keyword,emotion)
                # self.doBasicGensim(originfile,corpus)
                # self.doTWds(originfile,corpus)
                '''try:
                    os.mkdir('resources/wordclouds/notinclundingkeyword/' + keyword + '_' + emotion.lower())
                except Exception as e:
                    print(e)
                corpus=self.getCorpusTextFromRaw(raw_corpus)
                list_of_list_of_tokens = list(self.sent_to_words(corpus))
                list_of_list_of_tokens_no_keyword = [[tok for tok in l if tok != keyword] for
                                                       l in list_of_list_of_tokens]
                corpus=[" ".join(l) for l in list_of_list_of_tokens_no_keyword]
                try:
                    self.create_wordcloud(corpus, stopwords,path='resources/wordclouds/notinclundingkeyword/' + keyword + '_' + emotion.lower() + '_wordcloud.png')
                except ValueError:
                    None
                raw_corpus_by_nation=self.cluster_raw_corpus_by_nation(raw_corpus)
                for nat in raw_corpus_by_nation.keys():
                    print(nat)
                    raw_corpus_nat_text=self.getCorpusTextFromRaw(self.get_raw_corpus_nat(nat, raw_corpus_by_nation))
                    list_of_list_of_tokens = list(self.sent_to_words(raw_corpus_nat_text))
                    list_of_list_of_tokens_no_keyword = [[tok for tok in l if tok != keyword] for
                                                         l in list_of_list_of_tokens]
                    raw_corpus_nat_text = [" ".join(l) for l in list_of_list_of_tokens_no_keyword]
                    self.create_wordcloud(raw_corpus_nat_text, stopwords,
                                          path='resources/wordclouds/notinclundingkeyword/' + keyword + '_' + emotion.lower() + '/'+nat+'_wordcloud.png')'''
    def doKaggle(self,raw_corpus,stopwords,keyword,emotion):
        # https://www.kaggle.com/michaelcwang2/topic-modeling-for-hotel-review
        list_of_list_of_tokens = list(self.sent_to_words(raw_corpus))#tokenization+remove punctuation
        list_of_list_of_tokens_no_stopwords=[[tok for tok in l if tok not in stopwords and tok!=keyword] for l in list_of_list_of_tokens]
        #print(stopwords)
        if len(list_of_list_of_tokens)==0 or len(list_of_list_of_tokens_no_stopwords)==0:return
        # https://spacy.io/usage/processing-pipelines
        nlp = spacy.load('en', disable=['parser', 'ner'])
        #nlp =spacy.load("en_core_web_sm",disable=['parser', 'ner'])
        data_lemmatized = self.lemmatization(list_of_list_of_tokens_no_stopwords, nlp,#list of lists of tokens->list of strings
                                             allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        # NMF is able to use tf-idf
        #tfidf_vectorizer = TfidfVectorizer(stop_words='english')#Convert a collection of raw documents to a matrix of TF-IDF features. Equivalent to CountVectorizer followed by TfidfTransformer.
        #tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords)  # Convert a collection of raw documents to a matrix of TF-IDF features. Equivalent to CountVectorizer followed by TfidfTransformer.
        #tfidf = tfidf_vectorizer.fit_transform(data_lemmatized)#Learn vocabulary and idf, return term-document matrix.
        #tfidf_feature_names = tfidf_vectorizer.get_feature_names()#Array mapping from feature integer indices to feature name
        # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
        tf_vectorizer = CountVectorizer(analyzer='word',##############Convert a collection of text documents to a matrix of token counts
                                        min_df=10,  # minimum read occurences of a word
                                        stop_words=stopwords,  # remove stop words
                                        lowercase=True,  # convert all words to lowercase
                                        token_pattern='[a-zA-Z0-9]{2,}',  # num chars > 2
                                        )
        tf = tf_vectorizer.fit_transform(data_lemmatized)
        tf_feature_names = tf_vectorizer.get_feature_names()
        '''
        # Materialize the sparse data
        data_dense = tf.todense()#Return a dense matrix representation of this matrix.'''
        '''A NumPy matrix object with the same shape and containing the same data represented by the sparse matrix,
         with the requested memory order. If out was passed and was an array (rather than a numpy.matrix), 
         it will be filled with the appropriate values and returned wrapped in a numpy.matrix object that shares the same memory.'''
        '''# Compute Sparsicity = Percentage of Non-Zero cells
        print("Sparsicity: ", ((data_dense > 0).sum() / data_dense.size) * 100, "%")'''
        #random_state=If int, random_state is the seed used by the random number generator;
        # If RandomState instance, random_state is the random number generator;
        #alpha=Constant that multiplies the regularization terms. Set it to zero to have no regularization.
        #The regularization mixing parameter, with 0 <= l1_ratio <= 1.
        # For l1_ratio = 0 the penalty is an elementwise L2 penalty (aka Frobenius Norm).
        # For l1_ratio = 1 it is an elementwise L1 penalty. For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.
        #‘nndsvd’: Nonnegative Double Singular Value Decomposition (NNDSVD)

        #Find two non-negative matrices (W, H) whose product approximates the non- negative matrix X.
        # This factorization can be used for example for dimensionality reduction, source separation or topic extraction.

        # Run LDA
        # Build LDA Model
        '''lda_model = LatentDirichletAllocation(n_components=no_topics,  # Number of topics
                                              max_iter=10,  # Max learning iterations
                                              learning_method='batch',
                                              random_state=100,  # Random state
                                              batch_size=128,  # n docs in each learning iter
                                              evaluate_every=-1,
                                              # compute perplexity every n iters, default: Don't
                                              n_jobs=-1,  # Use all available CPUs
                                              )
        #Perplexity is another way to calculate the likelihood. It is defined as the reciprocal geometric mean
        # of the token likelihoods in the test corpus given the model:

        lda_output = lda_model.fit_transform(tf)
        # Log Likelyhood: Higher the better
        print("Log Likelihood: ", lda_model.score(tf))
        # Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
        print("Perplexity: ", lda_model.perplexity(tf))'''
        lda_model=LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
                          evaluate_every=-1, learning_decay=0.5,
                          learning_method='batch', learning_offset=10.0,
                          max_doc_update_iter=100, max_iter=10,
                          mean_change_tol=0.001, n_components=5, n_jobs=None,
                          perp_tol=0.1, random_state=None,
                          topic_word_prior=None,
                          verbose=0)
        '''best_lda_model=self.getBestLdaModel(tf)
        lda_model=best_lda_model
        print(lda_model)
        exit()'''
        '''
        # Get Log Likelyhoods from Grid Search Output
        gscore = model.cv_results_
        #cv_results_=A dict with keys as column headers and values as columns, that can be imported into a pandas DataFrame
        print(gscore['params'])
        print(gscore['mean_test_score'])
        # print([gscore['mean_test_score'][gscore['params'].index(v)] for v in gscore['params'] if v['learning_decay']==0.7])
        n_components = [5, 10, 15, 20]
        log_likelyhoods_5 = [gscore['mean_test_score'][gscore['params'].index(v)] for v in gscore['params'] if
                             v['learning_decay'] == 0.5]
        # print(log_likelyhoods_5)
        log_likelyhoods_7 = [gscore['mean_test_score'][gscore['params'].index(v)] for v in gscore['params'] if
                             v['learning_decay'] == 0.7]
        log_likelyhoods_9 = [gscore['mean_test_score'][gscore['params'].index(v)] for v in gscore['params'] if
                             v['learning_decay'] == 0.9]
        # import matplotlib as plt
        # %matplotlib inline
        # Show graph
        plt.figure(figsize=(12, 8))
        plt.plot(n_components, log_likelyhoods_5, label='0.5')
        plt.plot(n_components, log_likelyhoods_7, label='0.7')
        plt.plot(n_components, log_likelyhoods_9, label='0.9')
        plt.title("Choosing Optimal LDA Model")
        plt.xlabel("Num Topics")
        plt.ylabel("Log Likelyhood Scores")
        plt.legend(title='Learning decay', loc='best')
        plt.show()'''
        # Create Document - Topic Matrix
        #Transform data X according to the fitted model.
        #lda_output = lda_model.fit_transform(tfidf)#lda_ouput shape= (n_sentences,n_topics)
                                                    #lda_model shape after fit transform=(n_topics,n_words detected)
        lda_output = lda_model.fit_transform(tf)  # lda_ouput shape= (n_sentences,n_topics)
                                                        # lda_model shape after fit transform=(n_topics,n_words detected)
        # Log Likelyhood: Higher the better
        #print("Log Likelihood: ", lda_model.score(tfidf))
        # Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
        #print("Perplexity: ", lda_model.perplexity(tfidf))

        df_document_topic,df_document_topics,topicnames=self.buildpddataframedoctop(lda_model,data_lemmatized,lda_output)
        #print(df_document_topics.data)

        # start K-means analysis here
        print("doing kmeans")
        kmeans,clustering=self.kmeans(df_document_topic,num_trial_clusters=15)
        #em=self.EM(clustering,kmeans)
        kmeans.to_csv(path_or_buf='resources/topics/tf/clusterings/notincludingkeyword/withnegation/' + keyword + '_' + emotion + '.csv', sep='|')
        # Topic-Keyword Matrix
        df_topic_keywords = pd.DataFrame(
            lda_model.components_ / lda_model.components_.sum(axis=1)[:, np.newaxis])
        #Variational parameters for topic word distribution. Since the complete conditional for
        # topic word distribution is a Dirichlet, components_[i, j] can be viewed as pseudocount that
        # represents the number of times word j was assigned to topic i.
        #It can also be viewed as distribution over the words for each topic after normalization:

        # Assign Column and Index
        df_topic_keywords.columns = tf_vectorizer.get_feature_names()
        #df_topic_keywords.columns = tfidf_vectorizer.get_feature_names()
        df_topic_keywords.index = topicnames
        # View
        #df_topic_keywords.head(15)
        # Show top n keywords for each topic
        print("saving words per topic")
        topic_keywords = self.show_lda_topics(lda_model=lda_model, n_words=10,
                                              df_topic_keywords=df_topic_keywords)
        # Topic - Keywords Dataframe
        df_topic_keywords = pd.DataFrame(topic_keywords)
        df_topic_keywords.columns = ['Word ' + str(i) for i in range(df_topic_keywords.shape[1])]
        df_topic_keywords.index = ['Topic ' + str(i) for i in range(df_topic_keywords.shape[0])]
        #print(df_topic_keywords)
        pd.concat([df_document_topic, df_topic_keywords],sort=False).to_csv(path_or_buf='resources/topics/tf/notincludingkeyword/withnegation/' + keyword + '_' + emotion + '.csv',sep='|')

        # lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
        # Topic-Keyword matrix
        # tf_topic_keywords=pd.DataFrame(lda.components_/lda.components_.sum(axis=1)[:,np.newaxis])
        # Assign Columns and Index
        # tf_topic_keywords.columns=tf_feature_names
        # tf_topic_keywords.index=np.arange(0,no_topics)
        # print(tf_topic_keywords.head())

        '''no_topics = 15
        no_top_words = 8
        # Run NMF(Non-negative Matrix factorization)
        nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
        
        # display topic
        self.display_topics(nmf, tfidf_feature_names, no_top_words)'''
        # display lda through weighting
        '''def display_topics(feature_names, no_of_words):
        for topic_idx, topic in enumerate(tf_topic_keywords):
             print ("Topic %d:" % (topic_idx))
             print (" ".join([feature_names[i]
                             for i in topic.argsort()[:-no_of_words - 1:-1]]))'''
        # type(tf_feature_names)
        # tf_feature_array=np.asarray(tf_feature_names)
        # display_topics(lda, tf_feature_names, no_top_words)
        # doc_topic_dist = lda.transform(tf)
        # print(doc_topic_dist)
        # lda_perplexity=LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).perplexity(tf)
        # lda_score=LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).score(tf)
        # print("and lda score= "lda_score)
        # Importing Gensim
        # import matplotlib as plt
        # %matplotlib inline

    def doTWds(self,originfile,raw_corpus):
        list_of_list_of_tokens = list(self.sent_to_words(raw_corpus))
        #https://towardsdatascience.com/the-complete-guide-for-topics-extraction-in-python-a6aaa6cedbbc
        dictionary_LDA = corpora.Dictionary(list_of_list_of_tokens)
        dictionary_LDA.filter_extremes(no_below=0.01*len(list_of_list_of_tokens))
        corpus = [dictionary_LDA.doc2bow(list_of_tokens) for list_of_tokens in list_of_list_of_tokens]
        #lda_model = models.LdaModel(corpus, id2word=dictionary_LDA)
        num_topics = 20
        lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary_LDA, passes=4, alpha=[0.01] * num_topics, eta=[0.01] * len(dictionary_LDA.keys()))
        tops=(lda_model.show_topics(num_topics=num_topics, num_words=100, log=False, formatted=True))
        ratiotopiccorpuszero=lda_model[corpus[0]]  # corpus[0] means the first document.
        # print(lda_model[dictionary_LDA.doc2bow(['milk','extremely','fresh'])])

    def doBasicGensim(self,originfile,raw_corpus):
        list_of_list_of_tokens = list(self.sent_to_words(raw_corpus))
        # https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/gensim%20Quick%20Start.ipynb
        vectorrep,dictionary=self.getVectorRepresentation(list_of_list_of_tokens)
        tfidfmod=self.getTfIdfModel(vectorrep,dictionary)

    def buildpddataframedoctop(self,best_lda_model,data_lemmatized,lda_output):
        # column names
        topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_components)]
        # index names
        #docnames = ["Doc" + str(i) for i in range(len(data_lemmatized))]
        docnames = data_lemmatized
        # Make the pandas dataframe
        df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)
        # Get dominant topic for each document
        dominant_topic = np.argmax(df_document_topic.values, axis=1)
        df_document_topic['dominant_topic'] = dominant_topic
        # Styling
        # Apply Style
        # df_document_topics = df_document_topic.head(15).style.applymap(self.color_green).applymap(self.make_bold)
        #df_document_topics = df_document_topic.style.applymap(self.color_green).applymap(self.make_bold)
        df_document_topics=None
        return df_document_topic,df_document_topics,topicnames

    def kmeans(self,df_document_topic,num_trial_clusters=5):
        df_document_topic_col=(list(df_document_topic.columns._data))[:-1]
        df_document_topic_k = df_document_topic[df_document_topic_col]
        # print(df_document_topic_k.info())
        # Using the elbow method to find the optimal number of clusters
        opt_num_clusters_elbow=self.elbow(num_trial_clusters,df_document_topic_k.values)
        #opt_num_clusters_silhouette= self.silhouette(num_trial_clusters, df_document_topic_k.values)
        #opt_num_clusters=int((opt_num_clusters_elbow+opt_num_clusters_silhouette)/2)
        opt_num_clusters=opt_num_clusters_elbow
        kmeans = KMeans(n_clusters=opt_num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=20)
        clustering = kmeans.fit(df_document_topic_k)
        #kmeansprediction=kmeans.predict(df_document_topic_k)
        '''plt.figure(figsize=(12, 8))
        plt.scatter(df_document_topic_k['Topic0'].iloc[0:299], df_document_topic_k['Topic1'].iloc[0:299],
                    c=kmeans.predict(df_document_topic_k)[0:299], s=50, )#s : scalar or array_like, shape (n, ), optional The marker size in points**2. Default is rcParams['lines.markersize'] ** 2.
        #c : color, sequence, or sequence of color, optional
        #A sequence of color specifications of length n. or
        # A sequence of n numbers to be mapped to colors using cmap and norm.
        centers = clustering.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 2], c=[0, 1, 2, 3, 4], s=500, alpha=0.5)
        # plt.show()
        # print(centers)'''
        '''b = np.unique(clustering_labels)#Find the unique elements of an array.
        # Returns the sorted unique elements of an array
        for n in b:
            print("Clustering {}".format(n) + " has {} Hotel Review,".format(clustering_labels.tolist().count(n)))
        df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(
            name="Num Documents")
        print(df_topic_distribution)'''
        # pyLDAvis.enable_notebook()
        # panel = pyLDAvis.sklearn.prepare(best_lda_model, tf, tf_vectorizer, mds='tsne')  # no other mds function like tsne used.
        # print(panel)
        df_document_topic_k['cluster']=clustering.labels_
        df_document_topic_k=df_document_topic_k.sort_values(by=['cluster'])
        #valueslabels=np.append(values, clustering.labels_.reshape(clustering.labels_.shape[0],1), axis=1)
        return df_document_topic_k,clustering

    def getCorpusTextFromRaw(self, raw_corpus):
        rev_only = [r[2] for r in raw_corpus]
        return rev_only

    def getBestLdaModel(self,tf):
        # Define Search Param
        search_params = {'n_components': [5, 10, 15, 20], 'learning_decay': [.5, .7, .9]}
        # Init the Model
        lda = LatentDirichletAllocation()
        model = GridSearchCV(lda, param_grid=search_params)
        # GridSearchCV implements a “fit” and a “score” method.
        # It also implements “predict”, “predict_proba”, “decision_function”, “transform” and “inverse_transform” if they are implemented in the estimator used.

        # Do the Grid Search
        model.fit(tf)
        # Best Model
        best_lda_model = model.best_estimator_  # Estimator that was chosen by the search,
        # i.e. estimator which gave highest score (or smallest loss if specified) on the left out data

        # Model Parameters
        print("Best Model's Params: ", model.best_params_)
        # Log Likelihood Score
        print("Best Log Likelihood Score: ", model.best_score_)
        # Perplexity
        print("Model Perplexity: ", best_lda_model.perplexity(tf))
        return best_lda_model

    def elbow(self, num_trial_clusters, values):
        # Within-Cluster-Sum-of-Squares (WCSS)
        wcssK = []
        distancesK = []
        for i in range(1, num_trial_clusters+1):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=20)
            kmeans.fit(values)
            wcssK.append(kmeans.inertia_)  # Sum of squared distances of samples to their closest cluster center.
        # print(wcssK)
        p1 = Point(x_init=0, y_init=wcssK[0])
        p2 = Point(x_init=num_trial_clusters - 1, y_init=wcssK[num_trial_clusters - 1])
        for k in range(0, num_trial_clusters):
            p = Point(x_init=k + 1, y_init=wcssK[k])
            distancesK.append(p.distance_to_line(p1, p2))
        # print(distancesK)
        print("The maximum distance is ", max(distancesK),
              "at " + str(distancesK.index(max(distancesK)) + 1) + "th clustering")
        plt.figure(figsize=(12, 8))
        plt.plot(range(1, num_trial_clusters+1), wcssK)
        plt.plot(range(1, num_trial_clusters+1), distancesK)
        plt.title("The elbow method_Topic modeling for Hotel Review")
        plt.xlabel("The number of clusters")
        plt.ylabel("WCSS")
        plt.legend(['wcss', 'distance'], loc='upper right')
        # plt.show()
        return distancesK.index(max(distancesK)) + 1

    def silhouette(self, num_trial_clusters, values):
        silhouettes=[]
        for i in range(2, num_trial_clusters+1):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=20)
            clustering=kmeans.fit(values)
            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed clusters
            silhouette = silhouette_score(values, clustering.labels_)  # Compute the mean Silhouette Coefficient of all samples.
            silhouettes.append(silhouette)
            '''The Silhouette Coefficient is calculated using the mean intra-cluster distance (a) 
            and the mean nearest-cluster distance (b) for each sample. 
            The Silhouette Coefficient for a sample is (b - a) / max(a, b). 
            To clarify, b is the distance between a sample and the nearest cluster that the sample is not a part of. 
            Note that Silhouette Coefficient is only defined if number of labels is 2 <= n_labels <= n_samples - 1.'''
        return silhouettes.index(max(silhouettes)) + 1

    def computeVariances(self, medianvalues,clusteredvalues):
        variances=[]
        def ff(x,u=None):
            return np.power(x-u,2)
        for i in range(len(medianvalues)):
            medianvalue=medianvalues[i]
            values=clusteredvalues[i]
            variance=(np.sum(np.apply_along_axis(ff,1,values,u=medianvalue),axis=0))/len(values)
            variances.append(variance)
        return variances

    def EM(self,clustering,pddataframe):
        norms = []
        pdfs = []
        probscl=[]
        ww=[0]*len(np.unique(clustering.labels_))
        xsl = []
        xsa=np.array([])
        def expect(clusters):
            wws=[]
            probxs = np.zeros(xsa.shape)
            for i in clusters:
                n = norm(meanvalues[i], stddevs[i])
                norms.append(n)
                pdf=n.pdf(xsa)
                pdfs.append(pdf)
                probxs+=n.pdf(xsa)
            for i in clusters:
                w = (pdfs[i] * probscl[i]) / probxs
                wws.append(w)
            return wws
        def max(clusters):
            sumw=0
            for i in clusters:
                sumw+=np.sum(ww[i],axis=0)
            for i in clusters:
                w=ww[i]
                mu=np.sum(w*xsa,axis=0)/np.sum(w,axis=0)
                meanvalues[i]=mu
                stdv=np.sqrt(np.sum(w*(np.power((xsa-mu),2)),axis=0)/np.sum(w,axis=0))
                stddevs[i]=stdv
                prob=np.sum(w,axis=0)/sumw
                probscl[i]=prob

        totelementcount=len(clustering.labels_)
        meanvalues = clustering.cluster_centers_
        clusteredvalues = [[pddataframe.values[j][:-1] for j in range(len(clustering.labels_)) if clustering.labels_[j] == l]
                           for l in np.unique(clustering.labels_)]
        for xv in clusteredvalues:
            for x in xv:
                xsl.append(x)
        xsa = np.asarray(xsl)
        variances = self.computeVariances(meanvalues, clusteredvalues)
        stddevs = np.sqrt(variances)
        clusteredvalues = np.array(clusteredvalues)
        for j in range(clusteredvalues.shape[0]):
            clusteredvalues[j] = np.array(clusteredvalues[j])
            prob = len(clusteredvalues[j]) / totelementcount
            probscl.append(prob)
        trials=20
        likelihodd=0
        while(trials>0):
            wprec=copy.deepcopy(ww)
            ww=expect(np.unique(clustering.labels_))
            b=True
            for i in range(len(ww)):
                b=b and (np.array_equal(ww[i],wprec[i]))
                if (np.array_equal(ww[i],wprec[i])):
                    print("True")
            print(b)
            max(np.unique(clustering.labels_))
            p=np.zeros((totelementcount,clusteredvalues[0].shape[1]))
            for j in range(len(np.unique(clustering.labels_))):
                n = norm(meanvalues[i], stddevs[i])
                pdf = n.pdf(xsa)
                p+=(probscl[0]*pdf)
            computedlikelihodd=np.sum(np.prod(p,axis=0))
            if computedlikelihodd>=likelihodd:
                likelihodd=computedlikelihodd
                trials=20
            else:
                trials-=1
        results=np.zeros((totelementcount,len(np.unique(clustering.labels_))))
        for i in range(len(np.unique(clustering.labels_))):
            n = norm(meanvalues[i], stddevs[i])
            pdf = n.pdf(xsa)
        return results


class Point:
    def __init__(self, x_init, y_init):
        self.x = x_init
        self.y = y_init

    def shift(self, x, y):
        self.x += x
        self.y += y

    def __repr__(self):
        return "".join(["Point(", str(self.x), ",", str(self.y), ")"])

    def distance_to_line(self, p1, p2):
        x_diff = p2.x - p1.x
        y_diff = p2.y - p1.y
        num = abs(y_diff * self.x - x_diff * self.y + p2.x * p1.y - p2.y * p1.x)
        den = math.sqrt(y_diff ** 2 + x_diff ** 2)
        return num / den
class Gaussian:
    "Model univariate Gaussian"
    def __init__(self, mu, sigma):
        #mean and standard deviation
        self.mu = mu
        self.sigma = sigma

    #probability density function
    def pdf(self, x):
        "Probability of a data point given the current parameters"
        y=(1/(self.sigma*np.sqrt(2*np.pi)))*np.exp(((-(x-self.mu))/2*np.power(self.sigma,2)))
        return y