import csv

import nltk
from nltk.tokenize import word_tokenize

class wordwriter:
    @classmethod
    def write(cls,originfile):
        keywords = []
        f = open("booking_keywords.txt", "r")
        for line in f:
            keywords.append(line[:-1])  # important to have last line with \n
        f.close()
        for emotion in ["good","bad"]:
            print("doing " + emotion)
            for keyword in keywords:
                columnsrow = ["-----COUNTRY-----"]
                print("doing " + keyword)
                try:
                    f = open("csvs/binarymatrices/" + keyword + "_" + emotion + "_binary.csv", mode='w')
                    f.close()
                    with open("csvs/" + keyword + "_" + emotion + ".csv", mode='r') as csv_file:
                        reader = csv.reader(csv_file, delimiter='|', quotechar='"')
                        row_count = sum(1 for row in reader)
                        print("row count: " + str(row_count))
                    with open("csvs/" + keyword + "_" + emotion + ".csv", mode='r') as csv_file:
                        reader = csv.reader(csv_file, delimiter='|', quotechar='"')
                        k = 0
                        wordlists = []
                        for row in reader:
                            k += 1
                            wordlist = []
                            if k % 100 == 0:
                                print("row number now: " + str(k))

                            wordtok = word_tokenize(row[2])
                            for i in range(len(wordtok)):
                                if wordtok[i] == "n't":
                                    wordtok[i] = 'not'
                                if wordtok[i] == 'ca':
                                    wordtok[i] = 'can'
                            wordtok = [w for w in wordtok if w.isalpha()]  # remove punctuations)
                            postag = nltk.pos_tag(wordtok)
                            lensen = len(wordtok)
                            for i in range(lensen):
                                if wordtok[i] not in ['has', 'had', 'have', 'was', 'be', 'having','being','been', 'is', 'are', 'am',
                                                      'gonna'] and postag[i][1] in (
                                'RB', 'RBR', 'RBS', 'RP', 'UH', 'VB', 'VBD', 'VBN', 'VBP', 'VBZ'):
                                    wordlist.append([postag[i][0]])
                                    neighbours = []
                                    if i == 0:
                                        j = 0
                                        while j < lensen and j <= 2:
                                            neighbours.append(wordtok[i + j])
                                            j += 1
                                    elif i == lensen - 1:
                                        j = 0
                                        while j < lensen and j <= 2:
                                            neighbours.append(wordtok[i - j])
                                            j += 1
                                    elif i == 1:
                                        neighbours.append(wordtok[i - 1])
                                        j = 0
                                        while j < lensen - 1 and j <= 2:
                                            neighbours.append(wordtok[i + j])
                                            j += 1
                                    elif i == lensen - 2:
                                        neighbours.append(wordtok[i + 1])
                                        j = 0
                                        while j < lensen - 1 and j <= 2:
                                            neighbours.append(wordtok[i - j])
                                            j += 1
                                    else:
                                        j = 1
                                        neighbours.append(wordtok[i])
                                        while j <= 2:
                                            neighbours.append(wordtok[i + j])
                                            neighbours.append(wordtok[i - j])
                                            j += 1
                                    wordlist.append(neighbours)
                            wordlists.append([row[1], wordlist])
                            for word in wordlist:
                                if " ".join(word) not in columnsrow:
                                    columnsrow.append(" ".join(word))
                            f = open("csvs/binarymatrices/" + keyword + "_" + emotion + "_binary.csv", mode='w')
                            writer = csv.writer(f, delimiter='|')
                            writer.writerow(columnsrow)
                            f.close()
                        f = open("csvs/binarymatrices/" + keyword + "_" + emotion + "_binary.csv", mode='a')
                        writer = csv.writer(f, delimiter='|')
                        print("num sents: " + str(len(wordlists)))
                        i = 0
                        for wordlist in wordlists:
                            i += 1
                            if i % 100 == 0:
                                print(str(i))
                            binaryseq = [0] * len(columnsrow)
                            binaryseq[0] = wordlist[0]
                            for word in wordlist[1]:
                                if " ".join(word) in columnsrow:
                                    binaryseq[columnsrow.index(" ".join(word))] = 1
                            writer.writerow(binaryseq)
                        f.close()
                except Exception as e:
                    print(e)
        print("done")