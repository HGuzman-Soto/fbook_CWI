# Adapted from: https://github.com/siangooding/cwi_2018/blob/master/Populating%20Word%20Features.ipynb
# Populating word Features
import pandas as pd
import numpy
import string
import regex as re
import argparse
import json
import spacy
from pathlib import Path


# Load the data set that needs populating
##########################################################################################################

"""

And then, reorganize code so its not like a jupyter notebook

"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run feature extraction')
    parser.add_argument('--all', '-a', type=int, default=0)
    parser.add_argument('--wikipedia', '-w', type=int, default=0)
    parser.add_argument('--wikinews', '-i', type=int, default=0)
    parser.add_argument('--news', '-n', type=int, default=0)
    parser.add_argument('--dev', '-d', type=str, default=None)
    parser.add_argument('--old_dataset', '-old', type=str, default=None)
    parser.add_argument('--test', '-t', type=str, default=None)
    parser.add_argument('--german', '-ge', type=str, default=None)
    parser.add_argument('--spanish', '-sp', type=str, default=None)

    array = []
    args = parser.parse_args()
    if (args.all == 1):
        array = ['WikiNews_Train', 'WikiNews_Test', 'News_Train',
                 'News_Test', 'Wikipedia_Train', 'Wikipedia_Test']
    if (args.wikipedia == 1):
        array += 'Wikipedia_Test', 'Wikipedia_Train'
    if (args.wikinews == 1):
        array += 'WikiNews_Test', 'WikiNews_Train'
    if (args.news == 1):
        array += 'News_Test', 'News_Train'
    if (args.dev):
        array += ['WikiNews_Dev', 'Wikipedia_Dev', 'News_Dev']
    if (args.old_dataset):
        array += ['2016_test', '2016_train']
    elif (args.test):
        array = [args.test]

##########################################################################################################

for x in array:

    if (args.test):
        location = "testing_data/data_files/data/" + args.test + ".csv"
        data_frame = pd.read_csv(location, encoding='utf-8-sig')
        # data_frame = data_frame.astype(str)
    if (args.dev):
        location = 'dev_data/'+x+'.tsv'
        data_frame = pd.read_table(location, names=('ID', 'sentence', 'start_index', 'end_index', 'word', 'total_native',
                                                    'total_non_native', 'native_complex', 'non_native_complex', 'complex_binary', 'complex_probabilistic'), encoding='utf-8-sig')

    if (args.old_dataset):
        location = "training_data/" + x + '.txt'
        data_frame = pd.read_table(location, names=(
            'sentence', 'word', 'index', 'complex_binary'))
        data_frame['word'] = data_frame['word'].astype(str)
        data_frame['sentence'] = data_frame['sentence'].astype(str)

    if(args.all == 1):
        location = 'training_data/'+x+'.tsv'
        data_frame = pd.read_table(location, names=('ID', 'sentence', 'start_index', 'end_index', 'word', 'total_native',
                                                    'total_non_native', 'native_complex', 'non_native_complex', 'complex_binary', 'complex_probabilistic'), encoding='utf-8-sig')

    data_frame['sentence'] = data_frame['sentence'].apply(
        lambda x: x.replace("%", "percent"))
    data_frame['sentence'] = data_frame['sentence'].apply(
        lambda x: x.replace("’", "'"))

    data_frame['split'] = data_frame['word'].apply(lambda x: x.split())

    data_frame['count'] = data_frame['split'].apply(lambda x: len(x))

    # We create a table that contains only the words
    words = data_frame[data_frame['count'] == 1]

    word_set = words.word.str.lower().unique()

    word_set = pd.DataFrame(word_set)
    word_set.columns = ['word']


##########################################################################################################
    print("begin cleaning")
    # Cleaning function for words
    remove = string.punctuation
    remove = remove.replace("-", "")
    remove = remove.replace("'", "")  # don't remove apostraphies

    remove = remove + '“'
    remove = remove + '”'

    pattern = r"[{}]".format(remove)  # create the pattern
    word_set['word'] = word_set['word'].apply(
        lambda x: x.translate({ord(char): None for char in remove}))

    print("finish cleaning", "\n")

##########################################################################################################
    # Get character ngrams and probabilites

##########################################################################################################
    # if German

    if(args.german):

        # get wikipedia corpus frequency

        wiki_corpus = pd.read_csv("corpus/german/wikipedia_corpus.txt", delim_whitespace=True)
        word_parse_features['wikipedia_freq'] = word_parse_features['word'].apply(lambda x: int(
        wiki_corpus.loc[wiki_corpus.word == x, 'frequency']) if any(wiki_corpus.word == x) else 0)
        
        # get Lang8 learners corpus frequency *NEEDS FINISHED

        learner_corpus = pd.read_csv("corpus/learner_corpus.csv")
        word_parse_features['learner_corpus_freq'] = word_parse_features['word'].apply(lambda x: int(
        learner_corpus.loc[learner_corpus.word == x, 'frequency']) if any(learner_corpus.word == x) else 0)

        # get subtitles frequency

        subtitles_corpus = pd.read_csv("corpus/german/subtitles_corpus.csv")
        word_parse_features['subtitles_freq'] = word_parse_features['word'].apply(lambda x: int(
            subtitles_corpus.loc[subtitles_corpus.word == x, 'frequency']) if any(subtitles_corpus.word == x) else 0)

        # get Google Books unigram frequency **needs finished

        word_parse_features['google frequency'] = word_parse_features.apply(
            get_frequency, axis=1)

        # Apply function to get word length
        word_set['length'] = word_set['word'].apply(lambda x: len(x))

        # apply function to get vowel count
        word_set['vowels'] = word_set['word'].apply(
            lambda x: sum([x.count(y) for y in "aeiouäöü"]))

##########################################################################################################
    # if Spanish

    if(args.spanish):

        #wikipedia corpus word frequency
        #unable to find for now

        #Lang-8 word frequencies


        #subtitles frequencies
        subtitles_corpus = pd.read_csv("corpus/spanish/subtitlex_esp-2.csv")
        word_parse_features['subtitles_freq'] = word_parse_features['word'].apply(lambda x: int(
            subtitles_corpus.loc[subtitles_corpus.word == x, 'frequency']) if any(subtitles_corpus.word == x) else 0)
        #google books unigram

        #unigram frequency of target words from training data

        #character bigrams of target words from training data
        word_parse_features['google_char_bigram'] = word_parse_features['word'].apply(
            lambda x: char_bigram(x))

        #word length
        word_set['length'] = word_set['word'].apply(lambda x: len(x))

        #POS tag
        spacynlp = spacy.load("es_core_news_sm")
        posdoc = spacynlp(word_set['word'])

        word_parse_features['POS_tag'] = word_set['word'].apply(lambda x: posdoc[x].pos_)
        #Syllable Count

        #Vowel Count
        word_set['vowels'] = word_set['word'].apply(
            lambda x: sum([x.count(y) for y in "aeiouáéíóúü"]))
###########################################################################################################
    # function to obtain syablles for words
    from datamuse import datamuse
    api = datamuse.Datamuse()

    print("geting syllabels")

    def get_syllables(word):
        syllables = 0
        word_results = api.words(sp=word, max=1, md='psf')
        if len(word_results) > 0:
            word = word_results[0]["word"]
            syllables = int(word_results[0]["numSyllables"])
        print("# of syllables:", syllables)
        return syllables

    # Apply function to get syllables
    word_set['syllables'] = word_set['word'].apply(
        lambda x: get_syllables(x))

    # Apply function to get word length
    word_set['length'] = word_set['word'].apply(lambda x: len(x))

    # apply function to get vowel count
    word_set['vowels'] = word_set['word'].apply(
        lambda x: sum([x.count(y) for y in "aeiou"]))

    word_set['consonants'] = word_set['word'].apply(
        lambda x: sum([x.count(y) for y in "bcdfghjklmnpqrstvwxyz"]))

    # take words and merge with values first you will need to clean the phrase column
    words['original word'] = words['word']
    words['word'] = words['word'].str.lower()
    words['word'] = words['word'].apply(
        lambda x: x.translate({ord(char): None for char in remove}))

    word_features = pd.merge(words, word_set)

    print('Finished getting syllabels', "\n")


##########################################################################################################
    # Now parse
    import pycorenlp
    import pandas as pd
    from pycorenlp import StanfordCoreNLP
    print("start core")
    nlp = StanfordCoreNLP('http://localhost:9000')

    if (args.old_dataset):
        sentences = data_frame[['sentence']].copy()
    else:
        sentences = data_frame[['sentence', 'ID']].copy()

    sentences = sentences.drop_duplicates()

    print("end core")

##########################################################################################################
    print("start first token")

    def removefirsttoken(x):
        x = x.split(' ', 1)[1]
        return x

    if 'WikiNews' in x:
        sentences['clean sentence'] = sentences['sentence'].apply(
            lambda x: removefirsttoken(x))

    else:
        sentences['clean sentence'] = sentences['sentence']

    # sentences.to_csv("debugging/sentences_noparse.csv", index=False) debugging

    print("start end token")
##########################################################################################################
    # function to parse sentences
    print("start parse sentence")

    def parse(string):
        output = nlp.annotate(string, properties={
            'annotators': 'pos,depparse,ner',
            'outputFormat': 'json'
        })
        return output

    # apply parsing to sentences
    sentences['parse'] = sentences['clean sentence'].apply(lambda x: parse(x))

    word_parse_features = pd.merge(sentences, word_features)

    print("finish parsing sentence")
###########################################################################################################

    def get_pos(row):
        word = row['word']
        parse = row['parse']
        for i in range(len(parse['sentences'][0]['tokens'])):

            comp_word = parse['sentences'][0]['tokens'][i]['word']
            comp_word = comp_word.lower()
            comp_word = comp_word.translate(
                {ord(char): None for char in remove})

            if comp_word == word:
                return parse['sentences'][0]['tokens'][i]['pos']

###########################################################################################################
    def get_ner(row):
        word = row['word']
        parse = row['parse']
        # print(parse, "TEDFS")
        # print(parse['sentences'][0]['tokens'], "DOFISPFDSIU")
        # print(parse['sentences'][1]['tokens'], "DOFISPFDSIU")

        for i in range(len(parse['sentences'][0]['tokens'])):
            comp_word = parse['sentences'][0]['tokens'][i]['word']
            comp_word = comp_word.lower()
            comp_word = comp_word.translate(
                {ord(char): None for char in remove})
            print(parse['sentences'][0]['tokens'][i]['ner'])

            if comp_word == word:
                pass


###########################################################################################################


    def get_dep(row):
        number = 0
        word = row['word']
        parse = row['parse']
        for i in range(len(parse['sentences'][0]['basicDependencies'])):
            comp_word = parse['sentences'][0]['basicDependencies'][i]['governorGloss']
            comp_word = comp_word.lower()
            comp_word = comp_word.translate(
                {ord(char): None for char in remove})

            if comp_word == word:
                number += 1

        return number


##########################################################################################################

    # Function to get the proper lemma

    print("start tagging")
    from nltk.corpus import wordnet

    def get_wordnet_pos(treebank_tag):

        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    print("finish tagging", "\n")

##########################################################################################################

    print("start lemmatizing")
    from nltk.stem import WordNetLemmatizer
    wordnet_lemmatizer = WordNetLemmatizer()

    def lemmatiser(row):

        word = row['word']
        pos = row['pos']

        try:
            lemma = wordnet_lemmatizer.lemmatize(word, get_wordnet_pos(pos))
            return lemma
        except:
            try:
                lemma = wordnet_lemmatizer.lemmatize(word)
                return lemma
            except:
                print(word)

    print("finish lemmatizing", "\n")
##########################################################################################################

    mrc_features = pd.read_csv('corpus/MRC.csv')

##########################################################################################################

    def aoa(word):
        word = ''.join(word.split()).lower()
        try:
            df = mrc_features.loc[mrc_features['word'] == word.upper()]
            fvalue = df.iloc[0]['AOA']
            return fvalue
        except:
            return 0
##########################################################################################################

    def cnc(word):
        word = ''.join(word.split()).lower()
        try:
            df = mrc_features.loc[mrc_features['word'] == word.upper()]
            fvalue = df.iloc[0]['CNC']
            return fvalue
        except:
            return 0

##########################################################################################################

    def fam(word):
        word = ''.join(word.split()).lower()
        try:
            df = mrc_features.loc[mrc_features['word'] == word.upper()]
            fvalue = df.iloc[0]['FAM']
            return fvalue
        except:
            return 0

##########################################################################################################

    def img(word):
        word = ''.join(word.split()).lower()
        try:
            df = mrc_features.loc[mrc_features['word'] == word.upper()]
            fvalue = df.iloc[0]['IMG']
            return fvalue
        except:
            return 0

##########################################################################################################

    def phon(word):
        word = ''.join(word.split()).lower()
        try:
            df = mrc_features.loc[mrc_features['word'] == word.upper()]
            fvalue = df.iloc[0]['NPHN']
            return fvalue
        except:
            return 0

##########################################################################################################

    # functions using wordnet
    from nltk.corpus import wordnet

##########################################################################################################

    def synonyms(word):
        synonyms = 0
        try:
            results = wordnet.synsets(word)
            synonyms = len(results)
            return synonyms
        except:
            return synonyms

##########################################################################################################
    def holonyms(word):
        holonyms = 0
        try:
            results = wordnet.synsets(word)
            holonyms = len(results[0].holonyms())
            return holonyms
        except:
            return holonyms

##########################################################################################################
    def meronyms(word):
        meronyms = 0
        try:
            results = wordnet.synsets(word)
            meronyms = len(results[0].meronyms())
            return meronyms
        except:
            return meronyms


##########################################################################################################

    def hypernyms(word):
        hypernyms = 0
        try:
            results = wordnet.synsets(word)
            hypernyms = len(results[0].hypernyms())
            return hypernyms
        except:
            return hypernyms

##########################################################################################################

    def hyponyms(word):
        hyponyms = 0
        try:
            results = wordnet.synsets(word)
        except:
            return hyponyms
        try:
            hyponyms = len(results[0].hyponyms())
            return hyponyms
        except:
            return hyponyms


##########################################################################################################

    # CNC, KFCAT, FAM, KFSMP, KFFRQ, NPHN, T-LFRQ


    def CNC_fun(word):

        table = mrc_features[mrc_features['word'] == word.upper()]

        if len(table) > 0:

            CNC = table['CNC'].values[0]
            CNC = int(CNC)

            return CNC
        else:
            y = 0
            return y


##########################################################################################################

    def KFCAT_fun(word):

        table = mrc_features[mrc_features['word'] == word.upper()]

        if len(table) > 0:

            KFCAT = table['KFCAT'].values[0]
            KFCAT = int(KFCAT)

            return KFCAT
        else:
            y = 0
            return y

##########################################################################################################
    # get rid?
    def FAM_fun(word):

        table = mrc_features[mrc_features['word'] == word.upper()]

        if len(table) > 0:

            FAM = table['FAM'].values[0]
            FAM = int(FAM)

            return FAM
        else:
            y = 0
            return y


##########################################################################################################

    def KFSMP_fun(word):

        table = mrc_features[mrc_features['word'] == word.upper()]

        if len(table) > 0:

            KFSMP = table['KFSMP'].values[0]
            KFSMP = int(KFSMP)

            return KFSMP
        else:
            y = 0
            return y

##########################################################################################################

    def KFFRQ_fun(word):

        table = mrc_features[mrc_features['word'] == word.upper()]

        if len(table) > 0:

            KFFRQ = table['KFFRQ'].values[0]
            KFFRQ = int(KFFRQ)

            return KFFRQ
        else:
            y = 0
            return y

##########################################################################################################

    def NPHN_fun(word):

        table = mrc_features[mrc_features['word'] == word.upper()]
        if len(table) > 0:

            NPHN = table['NPHN'].values[0]
            NPHN = int(NPHN)

            return NPHN
        else:
            y = 0
            return y

##########################################################################################################

    def TLFRQ_fun(word):

        table = mrc_features[mrc_features['word'] == word.upper()]
        if len(table) > 0:

            TLFRQ = table['T-LFRQ'].values[0]
            TLFRQ = int(TLFRQ)

            return TLFRQ
        else:
            y = 0
            return y


##########################################################################################################

    # Convert tree bank tags to ones that are compatible w google


    def is_noun(tag):
        return tag in ['NN', 'NNS', 'NNP', 'NNPS']

    def is_verb(tag):
        return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

    def is_adverb(tag):
        return tag in ['RB', 'RBR', 'RBS']

    def is_adjective(tag):
        return tag in ['JJ', 'JJR', 'JJS']

    def penn_to_wn(tag):
        if is_adjective(tag):
            return wordnet.ADJ
        elif is_noun(tag):
            return wordnet.NOUN
        elif is_adverb(tag):
            return wordnet.ADV
        elif is_verb(tag):
            return wordnet.VERB
        return None

    def penn_to_google(tag):
        if is_adjective(tag):
            return 'adj'
        elif is_noun(tag):
            return 'n'
        elif is_adverb(tag):
            return 'adv'
        elif is_verb(tag):
            return 'v'
        return None

##########################################################################################################

    def get_frequency(row):
        nofreq = float(0.000000)
        word = row["word"]
        print("word:", word)
        word = str(word)
        tag = row["pos"]
        tag = penn_to_google(tag)

        try:
            word_results = api.words(sp=word, max=1, md='pf')
            tag_list = (word_results[0]['tags'][:-1])

            frequency = word_results[0]['tags'][-1][2:]

            frequency = float(frequency)

            if tag in tag_list:
                print("frequency_1:", frequency)
                return frequency
            else:
                lemma = row['lemma']
                try:
                    word_results = api.words(sp=lemma, max=1, md='pf')
                    tag_list = (word_results[0]['tags'][:-1])
                    frequency = word_results[0]['tags'][-1][2:]
                    frequency = float(frequency)

                    if tag in tag_list:
                        print("frequency_2:", frequency)
                        return frequency
                    else:
                        print("nofreq")
                        return nofreq
                except:
                    return nofreq

        except:

            return nofreq

##########################################################################################################

    # GET DEP AND POS NUMBER
    print("start get pos")
    word_parse_features['pos'] = word_parse_features.apply(get_pos, axis=1)
    print("end get pos")

    print("start get dep")

    word_parse_features['dep num'] = word_parse_features.apply(get_dep, axis=1)
    word_parse_features['ner'] = word_parse_features.apply(get_ner, axis=1)

    print("end get dep")

##########################################################################################################

    # To obtain word lemmas
    # Get Lemma
    word_parse_features['lemma'] = word_parse_features.apply(
        lemmatiser, axis=1)


##########################################################################################################

    # Apply function to get number of synonyms and hypernyms/hyponyms
    print("get syn, hyper, hypo")
    word_parse_features['synonyms'] = word_parse_features['lemma'].apply(
        lambda x: synonyms(x))
    word_parse_features['hypernyms'] = word_parse_features['lemma'].apply(
        lambda x: hypernyms(x))
    word_parse_features['hyponyms'] = word_parse_features['lemma'].apply(
        lambda x: hyponyms(x))
    word_parse_features['holonyms'] = word_parse_features['lemma'].apply(
        lambda x: holonyms(x))
    word_parse_features['meronyms'] = word_parse_features['lemma'].apply(
        lambda x: meronyms(x))
    print("end syn, hyper, hypo")

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

    print("start wikipedia corpus")
    wikipedia_corpus = pd.read_csv('corpus/wikipedia_corpus.csv')
    word_parse_features['wikipedia_freq'] = word_parse_features['word'].apply(
        lambda x: get_wiki(x))
    print("end wikipedia corpus")

##########################################################################################################
    print("start subtitles corpus, ogden, simple wiki, cald, learner, complex lexicon, and bnc corpus")
    subtitles_corpus = pd.read_csv("corpus/subtitles_corpus.csv")
    word_parse_features['subtitles_freq'] = word_parse_features['word'].apply(lambda x: int(
        subtitles_corpus.loc[subtitles_corpus.word == x, 'frequency']) if any(subtitles_corpus.word == x) else 0)

##########################################################################################################
    learner_corpus = pd.read_csv("corpus/learner_corpus.csv")
    word_parse_features['learner_corpus_freq'] = word_parse_features['word'].apply(lambda x: int(
        learner_corpus.loc[learner_corpus.word == x, 'frequency']) if any(learner_corpus.word == x) else 0)

##########################################################################################################
    word_complexity_lexicon = pd.read_csv(
        "corpus/lexicon.csv")
    word_parse_features['complex_lexicon'] = word_parse_features['word'].apply(lambda x: float(
        word_complexity_lexicon.loc[word_complexity_lexicon.word == x, 'score']) if any(word_complexity_lexicon.word == x) else 0)
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

    bnc_corpus = pd.read_csv("corpus/bnc_corpus.csv")

    word_parse_features['bnc_freq'] = word_parse_features['word'].apply(
        lambda x: get_bnc(x))


##########################################################################################################

    ogden = pd.read_table('binary-features/ogden.txt')
    word_parse_features['ogden'] = word_parse_features['lemma'].apply(
        lambda x: 1 if any(ogden.words == x) else 0)  # clean words
    print("end ogden")


##########################################################################################################

    simple_wiki = pd.read_csv('binary-features/Most_Frequent.csv')
    word_parse_features['simple_wiki'] = word_parse_features['lemma'].apply(
        lambda x: 1 if any(simple_wiki.a == x) else 0)  # clean words

##########################################################################################################

    # # Apply function to get the level from Cambridge Advanced Learner Dictionary
    cald = pd.read_csv('binary-features/CALD.csv')
    word_parse_features['cald'] = word_parse_features['word'].apply(
        lambda x: int(cald.loc[cald.Word == x, 'Level'].mean().round(0)) if any(cald.Word == x) else 0)

##########################################################################################################

    subimdb_500 = pd.read_csv('binary-features/subimbd_500.csv')
    word_parse_features['sub_imdb'] = word_parse_features['lemma'].apply(
        lambda x: 1 if any(subimdb_500.words == x) else 0)

##########################################################################################################

    print("end ogden, subimd, simple wiki, cald, learner, complex lexicon, and bnc corpus")

##########################################################################################################
    mrc_features = pd.read_csv('corpus/MRC.csv')

    print("get mrc cnc and img")
    word_parse_features['cnc'] = word_parse_features['lemma'].apply(
        lambda x: cnc(x))
    word_parse_features['img'] = word_parse_features['lemma'].apply(
        lambda x: img(x))
    word_parse_features['aoa'] = word_parse_features['lemma'].apply(
        lambda x: aoa(x))

    word_parse_features['fam'] = word_parse_features['lemma'].apply(
        lambda x: fam(x))

    print("end mrc cnc and img")


##########################################################################################################

    # Apply function for google freq
    print("get google frequency")
    word_parse_features['google frequency'] = word_parse_features.apply(
        get_frequency, axis=1)
    print("end google frequency")

##########################################################################################################

    print("start convert word and pos to string")
    word_parse_features['word'] = word_parse_features.word.astype(str)
    word_parse_features['pos'] = word_parse_features.pos.astype(str)
    print("end convert word and pos to string")


##########################################################################################################

    print('get rest of mrc')
    word_parse_features['KFCAT'] = word_parse_features['lemma'].apply(
        lambda x: KFCAT_fun(x))
    word_parse_features['KFSMP'] = word_parse_features['lemma'].apply(
        lambda x: KFSMP_fun(x))
    word_parse_features['KFFRQ'] = word_parse_features['lemma'].apply(
        lambda x: KFFRQ_fun(x))
    word_parse_features['NPHN'] = word_parse_features['lemma'].apply(
        lambda x: NPHN_fun(x))
    word_parse_features['TLFRQ'] = word_parse_features['lemma'].apply(
        lambda x: TLFRQ_fun(x))

    print('end rest of mrc')

##########################################################################################################

    word_parse_features['parse'] = word_parse_features.parse.astype(str)
    word_parse_features['split'] = word_parse_features['split'].astype(str)

    word_parse_features = word_parse_features.drop_duplicates()
    word_parse_features.to_pickle('features/'+x+'_allInfo')

    print(x)
