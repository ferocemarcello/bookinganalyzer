import csvwriter
import gensimlda
import gensimtutlsi

import lemm
import topicwriter
import wordwriter

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
    gensimlda.do()
    '''for t in ['tfidf','tf']:
        for k in ['notincludingkeyword','includingkeyword']:
            for n in ['withnegation','nonegation']:
                print('doing '+t+'-'+k+'-'+n)
                topicwriter.TopicWriter().do("booking_keywords.txt", tf=t, includingkeword=k,
                                             negation=n)'''