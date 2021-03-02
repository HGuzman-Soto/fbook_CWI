import pandas as pd
import csv
import dill
from collections import OrderedDict, defaultdict

# Get dataset from https://nlp.cs.nyu.edu/wikipedia-data/


def main(arg):
    if arg == "unigram":
        df = pd.read_csv('wp_1gram.txt', delim_whitespace=True,
                         quoting=csv.QUOTE_NONE)
    # elif arg == "bigram":
    #     df = pd.read_csv('wp_2gram.txt', delim_whitespace=True,
    #                     quoting=csv.QUOTE_NONE)
    #     dd = defaultdict(list)
    #     df.to_dict('records', into=dd)
    #     df.to_dict()

    nan_value = float("NaN")

    # casing
    df['word'] = df.word.str.lower()

    # remove punctuaction
    df['word'] = df.word.str.replace(
        '[^\w\s]', '')

    # remove numbers
    df['word'] = df.word.str.replace('\d+', '')

    # remove whitespace
    df['word'] = df.word.str.strip()
    df = df.dropna(subset=['word'])
    df['word'] = df.word[df.word.map(len) > 0]

    df['frequency'] = df.frequency.astype(int)
    df = df.dropna(subset=['word'])

    df = df[['word', 'frequency']]
    print(len(df))

    if(arg == 'unigram'):
        df.to_csv('wikipedia_corpus.csv', index=False)
    elif arg == "bigram":
        df.to_csv('wikipedia_bigram.csv', index=False)


df = pd.read_csv('wp_2gram.txt', delim_whitespace=True,
                 quoting=csv.QUOTE_NONE)
dd = defaultdict(list)
test = df.to_dict('records', into=dd)
dill.dump(test, open("wikipedia_bigram" + ".sav", 'rb'))
# main("bigram")
