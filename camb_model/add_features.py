import pandas as pd
import dill
import re
import math
from collections import Counter, defaultdict
from nltk import word_tokenize


"""
Script used to take pickle feature dataframes and add new features. This is done
as to not run feature extraction from scratch with existing datasets. The output 
is pickled dataframes
"""

wikipedia_corpus = pd.read_csv('corpus/wikipedia_corpus.csv')
bnc_corpus = pd.read_csv("corpus/bnc_corpus.csv")

##########################################################################################################
"""
    Wikipedia word may appear multiple times, we get the first instance. 
    We may have to sort frequency
    """


def get_wiki(word):
    df = wikipedia_corpus[wikipedia_corpus['word'] == word.lower()]
    if (len(df) > 0):

        wikipedia_freq = df['frequency'].values[0]

        wikipedia_freq = int(wikipedia_freq)

        return wikipedia_freq
    else:
        y = 0
        return y


def feat_wikipedia_corpus(word_parse_features):
    print("start wikipedia corpus")
    word_parse_features['wikipedia_freq'] = word_parse_features['word'].apply(
        lambda x: get_wiki(x))
    print("end wikipedia corpus")
    return word_parse_features

##########################################################################################################


def subtitle_corpus(word_parse_features):
    print("get subtitles")
    subtitles_corpus = pd.read_csv("corpus/subtitles_corpus.csv")
    word_parse_features['subtitles_freq'] = word_parse_features['word'].apply(lambda x: int(
        subtitles_corpus.loc[subtitles_corpus.word == x, 'frequency']) if any(subtitles_corpus.word == x) else 0)
    print("end subtitles")

    return word_parse_features
##########################################################################################################


def learner_corpus(word_parse_features):
    print("get learners")

    learner_corpus = pd.read_csv("corpus/learner_corpus.csv")
    word_parse_features['learner_corpus_freq'] = word_parse_features['word'].apply(lambda x: int(
        learner_corpus.loc[learner_corpus.word == x, 'frequency']) if any(learner_corpus.word == x) else 0)
    print("end learners")
    return word_parse_features
##########################################################################################################


def word_complexity(word_parse_features):
    print("get lexicon")
    word_complexity_lexicon = pd.read_csv(
        "corpus/lexicon.csv")
    word_parse_features['complex_lexicon'] = word_parse_features['word'].apply(lambda x: float(
        word_complexity_lexicon.loc[word_complexity_lexicon.word == x, 'score']) if any(word_complexity_lexicon.word == x) else 0)
    print("end lexicon")
    return word_parse_features
##########################################################################################################


"""
To do, so in the bnc corpus a word will appear multiple times. We will have to check (and convert)
its pos tag to line it up with the right frequency
"""


def get_bnc(word):
    df = bnc_corpus[bnc_corpus['word'] == word.lower()]
    if (len(df) > 0):

        bnc_freq = df['frequency'].values[0]

        bnc_freq = int(bnc_freq)

        return bnc_freq
    else:
        y = 0
        return y


def feat_bnc_corpus(word_parse_features):
    print("get bnc")

    word_parse_features['bnc_freq'] = word_parse_features['word'].apply(
        lambda x: get_bnc(x))

    print("end bnc")
    return word_parse_features


def vowels(word_parse_features):
    print("get vowels")
    word_parse_features['vowels'] = word_parse_features['word'].apply(
        lambda x: sum([x.count(y) for y in "aeiou"]))
    return word_parse_features

    print("end vowels")


def consonants(word_parse_features):
    print("get consonants")
    word_parse_features['consonants'] = word_parse_features['word'].apply(
        lambda x: sum([x.count(y) for y in "bcdfghjklmnpqrstvwxyz"]))
    return word_parse_features

##########################################################################################################


def holonyms(word):
    from nltk.corpus import wordnet
    holonyms = 0
    try:
        results = wordnet.synsets(word)
        holonyms = len(results[0].holonyms())
        return holonyms
    except:
        return holonyms

##########################################################################################################


def meronyms(word):
    from nltk.corpus import wordnet
    meronyms = 0
    try:
        results = wordnet.synsets(word)
        meronyms = len(results[0].meronyms())
        return meronyms
    except:
        return meronyms


##########################################################################################################
bigram_model = dill.load(
    open("lm/" + "simple_wikipedia" + ".sav", 'rb'))


def get_bigram_wiki(sentence, word, end_index):
    sentence = sentence.lower()
    sentence = re.sub(r"[^\w\d'\s]+", '', sentence)

    sentence_list = word_tokenize(sentence)

    # check if duplicates
    if (len(sentence_list) == len(set(sentence_list)) and word in sentence_list):
        word_index = sentence_list.index(word)
        prev_word = sentence_list[word_index-1]
        return bigram_model[word][prev_word]
    else:
        try:
            if end_index < len(sentence) // 2 and word in sentence_list:
                word_index = sentence_list.index(word)
            else:
                word_index = len(sentence_list) - 1 - \
                    sentence_list[::-1].index(word)
        except:
            try:
                sentence = re.sub(r'[^\w\s]', '', sentence)
                other_sentence_list = sentence.split()
                word_index = other_sentence_list.index(word)

            except:
                print("ERROR")
                return math.nan

        prev_word = sentence_list[word_index-1]

        print(bigram_model[word][prev_word])
        return bigram_model[word][prev_word]


def wiki_bigram(word_parse_features):

    bigram_prob = word_parse_features.apply(lambda x: get_bigram_wiki(
        x['clean sentence'], x.word, x['end_index']), axis=1)

    print(bigram_prob)
    return bigram_prob


##########################################################################################################

def get_bigrams_learners(sentence, word, end_index):
    sentence = sentence.lower()
    sentence = re.sub(r"[^\w\d'\s]+", '', sentence)

    sentence_list = word_tokenize(sentence)

    # check if duplicates
    if (len(sentence_list) == len(set(sentence_list)) and word in sentence_list):
        word_index = sentence_list.index(word)
        prev_word = sentence_list[word_index-1]
        return bigram_model[word][prev_word]
    else:
        try:
            if end_index < len(sentence) // 2 and word in sentence_list:
                word_index = sentence_list.index(word)
            else:
                word_index = len(sentence_list) - 1 - \
                    sentence_list[::-1].index(word)
        except:
            try:
                sentence = re.sub(r'[^\w\s]', '', sentence)
                other_sentence_list = sentence.split()
                word_index = other_sentence_list.index(word)

            except:
                print("ERROR")
                return math.nan

        prev_word = sentence_list[word_index-1]

        print(bigram_model[word][prev_word])
        return bigram_model[word][prev_word]


def learners_bigram(word_parse_features):
    learners_bigram = dill.load(
        open("lm/" + "learners" + ".sav", 'rb'))
    bigram_prob = word_parse_features.apply(lambda x: get_bigrams_learners(
        x['clean sentence'], x.word, x['end_index']), axis=1)

    print(bigram_prob)
    return bigram_prob


##########################################################################################################
array = ['News_Test_allInfo', 'News_Train_allInfo', 'WikiNews_Test_allInfo', 'News_Dev_allInfo', 'WikiNews_Dev_allInfo', 'Wikipedia_Dev_allInfo',
         'WikiNews_Train_allInfo', 'Wikipedia_Test_allInfo', 'Wikipedia_Train_allInfo']

#  '2016_train_allInfo']
#  '2016_test_allInfo']

for x in array:
    word_parse_features = pd.read_pickle('features/' + x)
    # word_parse_features = word_complexity(word_parse_features)
    # word_parse_features = subtitle_corpus(word_parse_features)
    # word_parse_features = feat_wikipedia_corpus(word_parse_features)
    # word_parse_features = learner_corpus(word_parse_features)
    # word_parse_features = feat_bnc_corpus(word_parse_features)

    # word_parse_features = vowels(word_parse_features)
    # word_parse_features['holonyms'] = word_parse_features['lemma'].apply(
    #     lambda x: holonyms(x))
    # word_parse_features['meronyms'] = word_parse_features['lemma'].apply(
    #     lambda x: meronyms(x))
    # word_parse_features = consonants(word_parse_features)
    word_parse_features['simple_wiki_bigrams'] = wiki_bigram(
        word_parse_features)
    # word_parse_features['learners_bigrams'] = learners_bigram(
    #     word_parse_features)

    word_parse_features.to_pickle(
        'features/' + x)
