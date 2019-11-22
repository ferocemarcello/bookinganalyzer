import bow_cluster_nation
import bow_cluster_country_tourist_hotel
import csvwriter
import frequence_difference_bow_creator
import gensimldamine
import gensimldatut
import gensimtutlsi
import indexmanager
import printtopicsforsentences
import lemm
import topicscoreanalyzer
import topicwriter
import wordwriter
import bagofwordanalyzer
if __name__ == '__main__':

    keywords=[]
    '''f = open("booking_keywords.txt", "r")
    for line in f:
        keywords.append(line[:-1])#important to have last line with \n
        #wordgetter.wordgetter.get_write_words([line[:-1]], filename=line[:-1] + '.txt')
    f.close()
    wordgetter.wordgetter.get_write_words(keywords, filename='booking_linked_keywords.txt')'''
    #viewmaker.viemaker.do("booking_keywords.txt")
    #lemm.lemm.stemlemmatizer("booking_keywords.txt")
    #csvwriter.do("booking_keywords.txt")
    #wordwriter.wordwriter.write("booking_keywords.txt")
    #gensimldamine.do('booking_keywords.txt')
    #printtopicsforsentences.saveweightedtopspersent('booking_keywords.txt')
    #topicscoreanalyzer.dividebynation('booking_keywords.txt')
    #bagofwordanalyzer.analyze('booking_keywords.txt')
    #bow_cluster_nation.do('booking_keywords.txt')
    #bow_cluster_country_tourist_hotel.do('booking_keywords.txt')
    indexmanager.build_country_indices()
    #frequence_difference_bow_creator.do('booking_keywords.txt')
    '''for t in ['tfidf','tf']:
        for k in ['notincludingkeyword','includingkeyword']:
            for n in ['withnegation','nonegation']:
                print('doing '+t+'-'+k+'-'+n)
                topicwriter.TopicWriter().do("booking_keywords.txt", tf=t, includingkeword=k,
                                             negation=n)'''