from nltk.tokenize import sent_tokenize, word_tokenize

from db import db_connection, db_operator
import csv
import spacy
from spacy_langdetect import LanguageDetector
nlp = spacy.load("en")
from textblob import TextBlob
import sys
import grammar_check
from gingerit.gingerit import GingerIt
from langid.langid import LanguageIdentifier, model
from spacy.tokens import Doc, Span
from googletrans import Translator

def custom_detection_function(spacy_object):
    # custom detection function should take a Spacy Doc or a
    assert isinstance(spacy_object, Doc) or isinstance(
        spacy_object, Span), "spacy_object must be a spacy Doc or Span object but it is a {}".format(type(spacy_object))
    detection = Translator().detect(spacy_object.text)
    return {'language':detection.lang, 'score':detection.confidence}

punctuation_list_space=[' ',',','.',';',':','!','"','?','-','_','(',')',"'"]

def do(originfile):
    db = db_connection()
    queryexecutor = db_operator(db)
    keywords = {}
    f = open(originfile, "r")
    for line in f:
        keyword = line[:-1]  # important to have last line with \n
        keywords[keyword] = []
        fs = open("subkeywords_booking/subkeywords_booking_cleaned/" + keyword + ".txt", "r")
        for linesub in fs:
            keywords[keyword].append(linesub[:-1])  # important to have last line with \n
        fs.close()
    f.close()
    #nlp.add_pipe(LanguageDetector(), name="language_detector", last=True)
    #nlp.add_pipe(LanguageDetector(language_detection_function=custom_detection_function), name="language_detector",last=True)
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    for emotion in ['Good', 'Bad']:
        print("begin " + emotion)
        for keyword in keywords.keys():
            if keyword in ['coffee','transportation','cleaning'] or emotion=='Bad':
                print(keyword)
                f=open('csvs/' + keyword + '_' + emotion.lower() + '.csv', mode='w')
                f.close()
                liketext = 'SELECT ReviewID, Country, ' + emotion + ' from masterthesis.reviews where '
                subkeywords = keywords[keyword]
                for subkey in subkeywords:
                    liketext += emotion + " LIKE '%" + subkey + "%' or "
                liketext = liketext[:-4]
                #liketext+=" limit 10000;"
                liketext += ";"
                db.connect()
                fields = queryexecutor.execute(query=liketext)
                db.disconnect()
                print("start analyzing sentences")
                toaddsents = []
                i=0
                print("number of reviews: "+str(len(fields)))
                for rew in fields:
                    i+=1
                    if i%10000==0:
                        print(str(i))
                    text = rew[2]
                    text = text.replace('\n', ' ')
                    text = text.replace('\t', ' ')
                    text = text.replace('\r', ' ')
                    text = text.replace('.', '. ')
                    text = text.replace(':', ': ')
                    text = text.replace(';', '; ')
                    text = text.replace(',', ', ')
                    '''lan=identifier.classify(text)[0]
                    acc=identifier.classify(text)[1]
                    if lan == 'en' and acc >= 0.9:
                        tokenized_text = sent_tokenize(text)
                        for sent in tokenized_text:
                            sentan = TextBlob(sent)
                            sent = (sentan.correct()).string
                            sentan=TextBlob(sent)
                            words = word_tokenize(sent)
                            skipsentence = True
                            for word in words:
                                if word.lower() in subkeywords:
                                    skipsentence = False
                                    break
                            if not skipsentence:
                                # tool = grammar_check.LanguageTool('en-GB')
                                # matches = tool.check(text)
                                sent = sent[0].upper() + sent[0 + 1:]
                                sent = sent.replace(' i ', ' I ')
                                # sentan.ngrams(3)
                                # sentan.sentiment_assessments
                                if sentan.polarity > 0.4 and sentan.subjectivity > 0.65:
                                    toaddsents.append([rew[0], rew[1], sent])'''
                    tokenized_text = sent_tokenize(text)
                    for sent in tokenized_text:
                        toaddsents.append([rew[0], rew[1], sent])
                print("start writing sentences")
                print("num sents: " + str(len(toaddsents)))
                i = 0
                csv_file = open('csvs/' + keyword + '_' + emotion.lower() + '.csv', mode='a')
                for sen in toaddsents:
                    i += 1
                    if i % 10000 == 0:
                        print(str(i))
                    writer = csv.writer(csv_file, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(sen)
                csv_file.close()
    print("done")