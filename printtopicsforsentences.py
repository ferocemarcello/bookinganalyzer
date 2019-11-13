import csv
import os

from gensim.models import LdaModel

import documentprocessor
import helper
from wordcloud import STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
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


def getStopwords(stopset):
    stopwords = set(STOPWORDS)
    stopwords.update(stopset)
    return stopwords
def saveweightedtopspersent(originfile):
    keywords = helper.getKeywords(originfile)
    for emotion in ['Good','Bad']:
        print("begin " + emotion)
        for keyword in keywords.keys():
            print(keyword)
            path='resources/gensim/noadj/not_cleaned/' + keyword + '_' + emotion.lower()+'/'+keyword+'_'+emotion.lower()
            try:
                lda = LdaModel.load(path)
                raw_corpus = helper.getRawCorpus(
                    csv_file=open('resources/csvs/' + keyword + '_' + emotion.lower() + '.csv', mode='r',
                                  encoding="utf8", newline='\n'), id_and_country=True, additionaldetails=True)
                stopwords = getStopwords(stopset)
                stwfromtfidf = list(TfidfVectorizer(stop_words='english').get_stop_words())
                stopwords = set(list(stopwords) + stwfromtfidf)
                for w in negationstopset:
                    stopwords.add(w)
                bow, dictionary, corpus,raw_corpus= documentprocessor.fullpreprocessrawcorpustobow(raw_corpus, stopwords,min_count_bigrams=20)

                if not os.path.exists('resources/gensim/noadj/outputtopsdocs/'):
                    os.makedirs('resources/gensim/noadj/outputtopsdocs/')
                if not os.path.exists('resources/gensim/noadj/outputtopsdocs/'+keyword+'_'+emotion.lower()+'/'):
                    os.makedirs('resources/gensim/noadj/outputtopsdocs/'+keyword+'_'+emotion.lower()+'/')
                csv_file = open(
                    'resources/gensim/noadj/outputtopsdocs/'+keyword+'_'+emotion.lower()+'/' + keyword + '_' + emotion.lower() + '.csv',
                    mode='w', encoding="utf8",
                    newline='\n')
                i = 0
                for val in lda.get_document_topics(bow):
                    s = [corpus[i], val]
                    writer = csv.writer(csv_file, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(raw_corpus[i]+s)
                    i += 1
                csv_file.close()
            except Exception as e:
                print(e)