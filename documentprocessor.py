import helper


def fullpreprocessrawcorpustobow(raw_corpus, stopwords, stwfromtfidf, negationstopset):
    corpus = helper.preprocessRawCorpus(raw_corpus, thresholdcountpernation=100)

    ###############################################################################
    # So we have a list of 1740 documents, where each document is a Unicode string.
    # If you're thinking about using your own corpus, then you need to make sure
    # that it's in the same format (list of Unicode strings) before proceeding
    # with the rest of this tutorial.
    #

    ###############################################################################
    # Pre-process and vectorize the documents
    # ---------------------------------------
    #
    # As part of preprocessing, we will:
    #
    # * Tokenize (split the documents into tokens).
    # * Lemmatize the tokens.
    # * Compute bigrams.
    # * Compute a bag-of-words representation of the data.
    #
    # First we tokenize the text using a regular expression tokenizer from NLTK. We
    # remove numeric tokens and tokens that are only a single character, as they
    # don't tend to be useful, and the dataset contains a lot of them.
    #
    # .. Important::
    #
    #    This tutorial uses the nltk library for preprocessing, although you can
    #    replace it with something else if you want.
    #

    # Tokenize the documents.
    from nltk.tokenize import RegexpTokenizer

    # Split the documents into tokens.
    stopwords = set(list(stopwords) + stwfromtfidf)
    for w in negationstopset:
        stopwords.add(w)

    tokenizer = RegexpTokenizer(r'\w+')
    print("starting tokenization")
    for idx in range(len(corpus)):
        corpus[idx] = corpus[idx].lower()  # Convert to lowercase.
        corpus[idx] = tokenizer.tokenize(corpus[idx])  # Split into words.
        corpus[idx] = [tok for tok in corpus[idx] if tok not in stopwords]
    # Remove numbers, but not words that contain numbers.
    corpus = [[token for token in doc if not token.isnumeric()] for doc in corpus]

    # Remove words that are shorter than 3 characters.
    corpus = [[token for token in doc if len(token) > 2] for doc in corpus]

    ###############################################################################
    # We use the WordNet lemmatizer from NLTK. A lemmatizer is preferred over a
    # stemmer in this case because it produces more readable words. Output that is
    # easy to read is very desirable in topic modelling.
    #

    # Lemmatize the documents.
    from nltk.stem.wordnet import WordNetLemmatizer

    print("starting lemmatization")
    lemmatizer = WordNetLemmatizer()
    corpus = [[lemmatizer.lemmatize(token) for token in doc] for doc in corpus]

    ###############################################################################
    # We find bigrams in the documents. Bigrams are sets of two adjacent words.
    # Using bigrams we can get phrases like "machine_learning" in our output
    # (spaces are replaced with underscores); without bigrams we would only get
    # "machine" and "learning".
    #
    # Note that in the code below, we find bigrams and then add them to the
    # original data, because we would like to keep the words "machine" and
    # "learning" as well as the bigram "machine_learning".
    #
    # .. Important::
    #     Computing n-grams of large dataset can be very computationally
    #     and memory intensive.
    #

    # Compute bigrams.
    from gensim.models import Phrases
    print("doing bigrams")
    # Add bigrams and trigrams to docs (only ones that appear 20 times or more).
    bigram = Phrases(corpus, min_count=20)
    for idx in range(len(corpus)):
        for token in bigram[corpus[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                corpus[idx].append(token)
    ###############################################################################
    # We remove rare words and common words based on their *document frequency*.
    # Below we remove words that appear in less than 20 documents or in more than
    # 50% of the documents. Consider trying to remove words only based on their
    # frequency, or maybe combining that with this approach.
    #

    # Remove rare and common tokens.
    from gensim.corpora import Dictionary

    # Create a dictionary representation of the documents.
    dictionary = Dictionary(corpus)

    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    # dictionary.filter_extremes(no_below=20, no_above=0.5)
    print("filtering extremes")
    dictionary.filter_extremes(no_below=0, no_above=0.5)

    ###############################################################################
    # Finally, we transform the documents to a vectorized form. We simply compute
    # the frequency of each word, including the bigrams.
    #

    # Bag-of-words representation of the documents.
    print("converting to vectors with doc2bow")
    bow = [dictionary.doc2bow(doc) for doc in corpus]
    return bow,dictionary,corpus