from nltk.stem import WordNetLemmatizer, PorterStemmer

class lemm:
    @staticmethod
    def stemlemmatizer(originfile):
        lemmtzer = WordNetLemmatizer()
        stemmtzer=PorterStemmer()
        keywords = {}
        f = open(originfile, "r")
        for line in f:
            keyword = line[:-1]  # important to have last line with \n
            keywords[keyword] = []
            fs = open("subkeywords_booking/subkeywords_booking_cleaned/" + keyword + ".txt", "r")
            fw = open(
                "subkeywords_booking/subkeywords_booking_cleaned/lemmatized/" + keyword + ".txt",
                mode='w+')
            fw.close()
            for linesub in fs:
                keywords[keyword].append(linesub[:-1])  # important to have last line with \n
            fs.close()
        f.close()

        for keyword in keywords:
            lemmatizedsubkeys = []
            for subkey in keywords[keyword]:
                lem=lemmtzer.lemmatize(subkey)
                stem=stemmtzer.stem(subkey)
                if stem not in lemmatizedsubkeys:
                    lemmatizedsubkeys.append(stem)
                if len(lem)>=len(stem):
                    if lem not in lemmatizedsubkeys:
                        lemmatizedsubkeys.append(lem)
            fw = open(
                "subkeywords_booking/subkeywords_booking_cleaned/lemmatized/" + keyword + ".txt",
                mode='a')
            for lemmatizedsubkey in lemmatizedsubkeys:
                fw.write(lemmatizedsubkey+'\n')
            fw.close()
