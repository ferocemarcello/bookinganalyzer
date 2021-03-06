import csv

from db import db_connection, db_operator


def getKeywords(originfile):
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
    return keywords
def getRawCorpus(csv_file, id_and_country=False, additionaldetails=False,limit=-1, offset=0):
    raw_corpus = []
    reader = csv.reader(csv_file, delimiter='|', quotechar='"')
    i = 0
    if additionaldetails:
        db = db_connection()
        queryexecutor = db_operator(db)
        db.connect()
        for row in reader:
            if i % 50000 == 0 and i!=0:
                print('reading sentence ' + str(i))
            i += 1
            if i>offset and i<=offset+limit or limit==-1:
                id=row[0]
                query = 'SELECT HotelNumber, FamilyType FROM masterthesis.reviews WHERE ReviewID='+id
                det = queryexecutor.execute(query=query)
                if len(det)<=0:
                    det = [('no_hotel_number'), ('no_family_type')]
                det = [det[0][0], det[0][1]]
                query = 'SELECT CountryID FROM masterthesis.hotels WHERE HotelNumber=' + str(det[0])
                hot = queryexecutor.execute(query=query)
                if len(hot)<=0:
                    hot=[('no_country',)]
                hot = [hot[0][0]]
                det=det+hot
                raw_corpus.append(row+det)
            if i>offset+limit and limit>0:
                break
        db.disconnect()
        return raw_corpus
    if id_and_country:
        for row in reader:
            if i % 50000 == 0 and i!=0:
                print('reading sentence ' + str(i))
            i += 1
            raw_corpus.append(row)
    else:
        for row in reader:
            if i % 50000 == 0 and i!=0:
                print('reading sentence ' + str(i))
            i += 1
            raw_corpus.append(row[2])
    csv_file.close()
    return raw_corpus
def cluster_raw_corpus_by_nation(raw_corpus):
    raw_corpus_by_nation={}
    for r in raw_corpus:
        if r[1]!='':
            if r[1] not in raw_corpus_by_nation.keys():
                raw_corpus_by_nation[r[1]]=[]
            raw_corpus_by_nation[r[1]].append(r)
    return raw_corpus_by_nation
def getCorpusTextFromRaw(raw_corpus,rewid=False):
    if rewid:
        text=[(r[0],r[2]) for r in raw_corpus]
    else:
        text = [r[2] for r in raw_corpus]
    return text

def preprocessRawCorpus(raw_corpus,thresholdcountpernation=100):
    raw_corpus_by_nation = cluster_raw_corpus_by_nation(raw_corpus)
    todeletenations = []
    for k in raw_corpus_by_nation.keys():
        if len(raw_corpus_by_nation[k]) < thresholdcountpernation:
            todeletenations.append(k)
    raw_corpus = [r for r in raw_corpus if r[1] not in todeletenations]
    corpus = getCorpusTextFromRaw(raw_corpus)
    return raw_corpus,corpus