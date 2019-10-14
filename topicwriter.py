import csv
import math

import matplotlib

import helper
import fasttext
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
import pyLDAvis as pyLDAvis
import pyLDAvis.sklearn

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
         "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
         "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
         "should", "now"])
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
        stopwords.update(["Nan", "Negative", "etc"])
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
            print(" ".join([feature_names[i]
                            for i in topic.argsort()[:-no_top_words - 1:-1]]))
    def show_lda_topics(self,lda_model=None, n_words=20,df_topic_keywords=None):
        keywords = np.array(df_topic_keywords.columns)
        topic_keywords = []
        for topic_weights in lda_model.components_:
            top_keyword_locs = (-topic_weights).argsort()[:n_words]
            topic_keywords.append(keywords.take(top_keyword_locs))
        return topic_keywords
    def create_wordcloud(self, text="",stopwords=[],display=False, save=False, path=""):
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

    def get_corpus_from_raw_corpus_by_nation(self, nat, raw_corpus_by_nation):
        reviews=raw_corpus_by_nation[nat]
        rev_only=[r[2] for r in reviews]
        return rev_only
    def do(self,originfile):
        keywords = helper.getKeywords(originfile)
        # Create stopword list:
        stopwords = self.getStopwords(self.stopset)
        for emotion in ['Good', 'Bad']:
            print("begin " + emotion)
            for keyword in keywords.keys():
                print(keyword)
                raw_corpus = self.getRawCorpus(
                    csv_file=open('resources/csvs/' + keyword + '_' + emotion.lower() + '.csv', mode='r',
                                  encoding="utf8", newline='\n'),all=True)
                raw_corpus_by_nation=self.cluster_raw_corpus_by_nation(raw_corpus)
                raw_corpus_text = self.getRawCorpus(
                    csv_file=open('resources/csvs/' + keyword + '_' + emotion.lower() + '.csv', mode='r',
                                  encoding="utf8", newline='\n'), all=False)
                # Generate a word cloud image
                text = " ".join(review for review in raw_corpus_text)
                try:
                    self.create_wordcloud(text=text, stopwords=stopwords, display=False, save=True,
                                      path='resources/wordclouds/' + keyword + '_' + emotion.lower() + '_wordcloud.png')
                except ValueError:
                    print("0 words in csv")
                for nat in raw_corpus_by_nation.keys():
                    print(nat)
                    raw_corpus_nat_text=self.get_corpus_from_raw_corpus_by_nation(nat,raw_corpus_by_nation)
                    #self.doBasicGensim(originfile,raw_corpus_nat_text)
                    #self.doTWds(originfile,raw_corpus_nat_text)
                    #self.doKaggle(originfile,raw_corpus_nat_text,keyword,emotion)

                    # Generate a word cloud image
                    text = " ".join(review for review in raw_corpus_nat_text)
                    self.create_wordcloud(text=text, stopwords=stopwords, display=False, save=True,
                                          path='resources/wordclouds/' + keyword + '_' + emotion.lower() + '_'+nat+'_wordcloud.png')
    def doKaggle(self,originfile,raw_corpus,keyword,emotion):
        list_of_list_of_tokens = list(self.sent_to_words(raw_corpus))
        # https://www.kaggle.com/michaelcwang2/topic-modeling-for-hotel-review
        nationalitiescount = self.getNationsCountDict(
            csv_file=open('resources/csvs/' + keyword + '_' + emotion.lower() + '.csv', mode='r',
                          encoding="utf8", newline='\n'))
        # Create stopword list:
        stopwords = self.getStopwords(self.stopset)
        # Generate a word cloud image
        text = " ".join(review for review in raw_corpus)
        self.create_wordcloud(text=text, stopwords=stopwords, display=False, save=False,
                              path='resources/wordclouds' + keyword + '_' + emotion.lower() + '_wordcloud.png')

        # https://spacy.io/usage/processing-pipelines
        nlp = spacy.load('en', disable=['parser', 'ner'])
        data_lemmatized = self.lemmatization(list_of_list_of_tokens, nlp,
                                             allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        # NMF is able to use tf-idf
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf = tfidf_vectorizer.fit_transform(data_lemmatized)
        tfidf_feature_names = tfidf_vectorizer.get_feature_names()
        # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
        tf_vectorizer = CountVectorizer(analyzer='word',
                                        min_df=10,  # minimum read occurences of a word
                                        stop_words='english',  # remove stop words
                                        lowercase=True,  # convert all words to lowercase
                                        token_pattern='[a-zA-Z0-9]{2,}',  # num chars > 2
                                        )
        tf = tf_vectorizer.fit_transform(data_lemmatized)
        tf_feature_names = tf_vectorizer.get_feature_names()
        # Materialize the sparse data
        data_dense = tf.todense()
        # Compute Sparsicity = Percentage of Non-Zero cells
        print("Sparsicity: ", ((data_dense > 0).sum() / data_dense.size) * 100, "%")
        no_topics = 15
        # Run NMF(Non-negative Matrix factorization)
        nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
        # Run LDA
        # Build LDA Model
        lda_model = LatentDirichletAllocation(n_components=no_topics,  # Number of topics
                                              max_iter=10,  # Max learning iterations
                                              learning_method='batch',
                                              random_state=100,  # Random state
                                              batch_size=128,  # n docs in each learning iter
                                              evaluate_every=-1,
                                              # compute perplexity every n iters, default: Don't
                                              n_jobs=-1,  # Use all available CPUs
                                              )
        lda_output = lda_model.fit_transform(tf)
        # Log Likelyhood: Higher the bette
        print("Log Likelihood: ", lda_model.score(tf))
        # Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
        print("Perplexity: ", lda_model.perplexity(tf))
        # Define Search Param
        search_params = {'n_components': [5, 10, 15, 20], 'learning_decay': [.5, .7, .9]}
        # Init the Model
        lda = LatentDirichletAllocation()
        model = GridSearchCV(lda, param_grid=search_params)
        # Do the Grid Search
        model.fit(tf)
        # Best Model
        best_lda_model = model.best_estimator_
        # Model Parameters
        print("Best Model's Params: ", model.best_params_)
        # Log Likelihood Score
        print("Best Log Likelihood Score: ", model.best_score_)
        # Perplexity
        print("Model Perplexity: ", best_lda_model.perplexity(tf))
        # Get Log Likelyhoods from Grid Search Output
        gscore = model.fit(tf).cv_results_
        print(gscore)
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
        # plt.show()
        # Create Document - Topic Matrix
        lda_output = best_lda_model.transform(tf)
        # column names
        topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_components)]
        # index names
        docnames = ["Doc" + str(i) for i in range(len(data_lemmatized))]
        # Make the pandas dataframe
        df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)
        # Get dominant topic for each document
        dominant_topic = np.argmax(df_document_topic.values, axis=1)
        df_document_topic['dominant_topic'] = dominant_topic
        # Styling
        # Apply Style
        df_document_topics = df_document_topic.head(15).style.applymap(self.color_green).applymap(
            self.make_bold)
        print(df_document_topics.data)
        print(df_document_topic.info())
        # start K-means analysis here
        df_document_topic_k = df_document_topic[["Topic0", "Topic1", "Topic2", "Topic3", "Topic4"]]
        print(df_document_topic_k.info())
        K = np.array(df_document_topic_k)
        # Using the elbow method to find the optimal number of clusters
        wcssK = []
        distancesK = []
        j = 15
        for i in range(1, j):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=20)
            kmeans.fit(df_document_topic_k)
            wcssK.append(kmeans.inertia_)
        print(wcssK)
        for k in range(1, j):
            p1 = Point(x_init=0, y_init=wcssK[0])
            p2 = Point(x_init=j - 1, y_init=wcssK[j - 2])
            p = Point(x_init=k - 1, y_init=wcssK[k - 2])
            distancesK.append(p.distance_to_line(p1, p2))
        print(distancesK)
        print("The maximum distance is ", max(distancesK),
              "at {}th clustering".format(distancesK.index(max(distancesK))))
        plt.figure(figsize=(12, 8))
        plt.plot(range(1, j), wcssK)
        plt.plot(range(1, j), distancesK)
        plt.title("The elbow method_Topic modeling for Hotel Review")
        plt.xlabel("The number of clusters")
        plt.ylabel("WCSS")
        plt.legend(['wcss', 'distance'], loc='upper right')
        # plt.show()
        kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=20)
        a = kmeans.fit(df_document_topic_k).labels_
        print(a[0:99])
        print(kmeans.predict(df_document_topic_k)[0:299])
        plt.figure(figsize=(12, 8))
        plt.scatter(df_document_topic_k['Topic0'].iloc[0:299], df_document_topic_k['Topic1'].iloc[0:299],
                    c=kmeans.predict(df_document_topic_k)[0:299], s=50, )
        centers = kmeans.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 2], c=[0, 1, 2, 3, 4], s=500, alpha=0.5)
        # plt.show()
        print(centers)
        b = np.unique(a)
        for n in b:
            print("Clustering {}".format(n) + " has {} Hotel Review,".format(a.tolist().count(n)))
        range_n_clusters = [3, 4, 5, 6, 7]
        for nth_clusters in range_n_clusters:
            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(df_document_topic_k, kmeans.fit_predict(df_document_topic_k))
            print("For n_clusters =", nth_clusters,
                  "The average silhouette_score is :", silhouette_avg)
        df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(
            name="Num Documents")
        print(df_topic_distribution)
        # pyLDAvis.enable_notebook()
        # panel = pyLDAvis.sklearn.prepare(best_lda_model, tf, tf_vectorizer, mds='tsne')  # no other mds function like tsne used.
        # print(panel)
        # Topic-Keyword Matrix
        df_topic_keywords = pd.DataFrame(
            best_lda_model.components_ / best_lda_model.components_.sum(axis=1)[:, np.newaxis])
        # Assign Column and Index
        df_topic_keywords.columns = tf_vectorizer.get_feature_names()
        df_topic_keywords.index = topicnames
        # View
        df_topic_keywords.head(15)
        # Show top n keywords for each topic
        topic_keywords = self.show_lda_topics(lda_model=best_lda_model, n_words=15,
                                              df_topic_keywords=df_topic_keywords)
        # Topic - Keywords Dataframe
        df_topic_keywords = pd.DataFrame(topic_keywords)
        df_topic_keywords.columns = ['Word ' + str(i) for i in range(df_topic_keywords.shape[1])]
        df_topic_keywords.index = ['Topic ' + str(i) for i in range(df_topic_keywords.shape[0])]
        print(df_topic_keywords)
        # lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
        # Topic-Keyword matrix
        # tf_topic_keywords=pd.DataFrame(lda.components_/lda.components_.sum(axis=1)[:,np.newaxis])
        no_top_words = 8
        # Assign Columns and Index
        # tf_topic_keywords.columns=tf_feature_names
        # tf_topic_keywords.index=np.arange(0,no_topics)
        # print(tf_topic_keywords.head())
        # display topic
        self.display_topics(nmf, tfidf_feature_names, no_top_words)
        # display lda through weighting
        # def display_topics(feature_names, no_of_words):
        # for topic_idx, topic in enumerate(tf_topic_keywords):
        #     print ("Topic %d:" % (topic_idx))
        #     print (" ".join([feature_names[i]
        #                     for i in topic.argsort()[:-no_of_words - 1:-1]]))
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