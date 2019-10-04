from nltk import word_tokenize
from nltk.corpus import wordnet
class wordgetter:

    @staticmethod
    def get_write_words(keywordlist, filename='booking_keywords.txt'):

        destfilename='subkeywords_booking/' + filename
        f = open(destfilename, "w")
        f.close()

        # Synset: It is also called as synonym set or collection of synonym words.
        linkedwords = dict()
        linkedwords["synonims"] = []
        linkedwords["antonyms"] = []
        linkedwords["hypernyms"] = []
        linkedwords["hyponyms"] = []
        linkedwords["substance_holonyms"] = []
        linkedwords["part_holonyms"] = []
        linkedwords["member_holonyms"] = []
        linkedwords["part_meronyms"] = []
        linkedwords["member_meronyms"] = []
        linkedwords["substance_meronyms"] = []

        for keyword in keywordlist:
            for syn in wordnet.synsets(keyword):
                linkedwords["synonims"].append(syn)

                for l in syn.lemmas():
                    if len(l.antonyms()) > 0:#Antonyms are words that have contrasting, or opposite, meanings.
                        linkedwords["antonyms"].append(l.antonyms()[0].name())

                if len(syn.hypernyms()) > 0:  # a word with a broad meaning constituting a category into which words
                    # with more specific meanings fall; a superordinate.For example, colour is a hypernym of red.
                    for hyp in syn.hypernyms():
                        linkedwords["hypernyms"].append(hyp)

                if len(syn.hyponyms()) > 0:  # a word of more specific meaning than a general or superordinate
                    # term applicable to it. For example, spoon is a hyponym of cutlery.
                    for hyp in syn.hyponyms():
                        linkedwords["hyponyms"].append(hyp)

                if len(
                        syn.member_holonyms()) > 0:  # A term that denotes a whole, a part of which is denoted by a second term.
                    # The word "face" is a holonym of the word "eye".
                    for hol in syn.member_holonyms():
                        linkedwords["member_holonyms"].append(hol)

                if len(
                        syn.part_holonyms()) > 0:  # A term that denotes a whole, a part of which is denoted by a second term.
                    # The word "face" is a holonym of the word "eye".
                    for hol in syn.part_holonyms():
                        linkedwords["part_holonyms"].append(hol)

                if len(
                        syn.substance_holonyms()) > 0:  # A term that denotes a whole, a part of which is denoted by a second term.
                    # The word "face" is a holonym of the word "eye".
                    for hol in syn.substance_holonyms():
                        linkedwords["substance_holonyms"].append(hol)

                if len(syn.part_meronyms()) > 0:
                    # a term which denotes part of something but which is used to refer
                    # to the whole of it, e.g. faces when used to mean people in I see several familiar faces present.
                    for mer in syn.part_meronyms():
                        linkedwords["part_meronyms"].append(mer)

                if len(syn.member_meronyms()) > 0:
                    # a term which denotes part of something but which is used to refer
                    # to the whole of it, e.g. faces when used to mean people in I see several familiar faces present.
                    for mer in syn.member_meronyms():
                        linkedwords["member_meronyms"].append(mer)

                if len(syn.substance_meronyms()) > 0:
                    # a term which denotes part of something but which is used to refer
                    # to the whole of it, e.g. faces when used to mean people in I see several familiar faces present.
                    for mer in syn.substance_meronyms():
                        linkedwords["substance_meronyms"].append(mer)

                punctuation_list = ['.', '-', '_', ',', ';', ':', '/', '\\', '*']

                for key in linkedwords.keys():
                    for word in linkedwords[key]:
                        if key == 'antonyms':
                            with open(destfilename) as fr:
                                repword=word
                                for punct in punctuation_list:
                                    if punct in word:
                                        repword = word.replace(punct, ' ')
                                tokenized_word = word_tokenize(repword)  # word tokenization
                                if word not in tokenized_word:
                                    tokenized_word.append(word)
                                for tok_word in tokenized_word:
                                    if tok_word + '\n' not in fr.read():
                                        f = open(destfilename, "a")
                                        f.write(tok_word + '\n')
                                        f.close()
                            fr.close()
                        else:
                            for name in word.lemma_names():
                                with open(destfilename) as fr:
                                    repname=name
                                    for punct in punctuation_list:
                                        if punct in name:
                                            repname=name.replace(punct,' ')
                                    tokenized_name = word_tokenize(repname)  # word tokenization
                                    if name not in tokenized_name:
                                        tokenized_name.append(name)
                                    for tok_nam in tokenized_name:
                                        if tok_nam + '\n' not in fr.read():
                                            f = open(destfilename, "a")
                                            f.write(tok_nam + '\n')
                                            f.close()
                                fr.close()
                    linkedwords[key] = []