# code courtesy of https://nlpforhackers.io/language-models/
import pandas as pd
import pickle
import dill
from nltk import bigrams, trigrams
from nltk.tokenize import sent_tokenize

from collections import Counter, defaultdict


def create_model(corpus_df, corpus_name):
    # Create a placeholder for model
    model = defaultdict(lambda: defaultdict(lambda: 0))

    # Count frequency of co-occurance
    i = 0
    for sentence in corpus_df.sentence:
        for w1, w2 in bigrams(sentence, pad_right=False, pad_left=False):
            model[(w1)][w2] += 1
            if i % 1000 == 1:
                print(w1, w2)
            i += 1

    # Let's transform the counts to probabilities
    for w1_w2 in model:
        total_count = float(sum(model[w1_w2].values()))
        for w3 in model[w1_w2]:
            model[w1_w2][w3] /= total_count

    dill.dump(model, open("lm/" + corpus_name + ".sav", 'wb'))


def simple_wiki():
    df_wiki = pd.read_table('one_off_scripts/wiki.simple', names=['sentence'])

    # casing
    df_wiki['sentence'] = df_wiki.sentence.str.lower()

    # remove punctuaction
    df_wiki['sentence'] = df_wiki.sentence.str.replace(
        '[^\w\s]', '')

    df_wiki['sentence'] = '<s> ' + df_wiki['sentence'] + ' </s>'

    # get rid of this line if we want char-ngrams
    df_wiki['sentence'] = df_wiki.sentence.str.strip().str.split()

    create_model(df_wiki, "simple_wikipedia")


def get_learner_data():
    # get dataset from https://sites.google.com/site/naistlang8corpora/home/readme-en

    fields = ['num_corrections', 'serial_num',
              'url', 'sentence_num', 'learner_eng', 'correction']
    df_test = pd.read_table("one_off_scripts/entries.test", sep=",",
                            usecols=fields)

    df_train = pd.read_table("one_off_scripts/entries.train", sep=",",
                             usecols=fields)

    df_train = df_train['learner_eng']
    df_test = df_test['learner_eng']

    merged_df = pd.concat([df_test, df_train], axis=0)
    merged_df = merged_df.astype(str)
    return merged_df


def learner():
    df_learner = get_learner_data()
    # casing
    df_learner['sentence'] = df_learner.str.lower()

    # remove punctuaction
    df_learner['sentence'] = df_learner.sentence.str.replace(
        '[^\w\s]', '')

    # add start and end of sentence tokens
    df_learner['sentence'] = '<s>' + df_learner['sentence'] + '</s>'

    # get rid of this line if we want char-ngrams
    df_learner['sentence'] = df_learner.sentence.str.strip().str.split()

    create_model(df_learner, "learners")

# simple_wiki()
# loaded_model = dill.load(
#         open("lm/" + "simple_wikipedia" + ".sav", 'rb'))


# print(loaded_model['tuberculate'])
# print(loaded_model['test'])
learner()
simple_wiki()
