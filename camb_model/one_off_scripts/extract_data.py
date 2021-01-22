import pandas as pd

import re
import collections
import string
from collections import Counter


"""For simple wikipedia

1) Convert text file to csv, set columns to [word,paragraph,sentence]
2) Read that csv file to pandas again
3) Get the top 6,386 words

https://stackoverflow.com/questions/29903025/count-most-frequent-100-words-from-sentences-in-dataframe-pandas


Might need to do some pre-processing


Issue: When converting series to dataframe, the index column is made when converting to csv
I just manually label the first colummn as word

"""
from nltk.corpus import stopwords
stop = stopwords.words('english')


def simple_wiki():

    df_wiki = pd.read_table('wiki.simple', names=['sentence'])

    # casing
    df_wiki['sentence'] = df_wiki.sentence.str.lower()
    # remove stop words
    df_wiki['sentence'] = df_wiki.sentence.apply(lambda x: ' '.join(
        [word for word in x.split() if word not in (stop)]))

    # remove punctuaction
    df_wiki['sentence'] = df_wiki.sentence.str.replace('[^\w\s]', '')
    # print(df_wiki)

    # remove numbers
    df_wiki['sentence'] = df_wiki.sentence.str.replace('\d+', '')
    df_wiki['sentence'] = df_wiki.sentence.str.strip()

    series_top = pd.Series(
        " ".join((df_wiki.sentence).str.lower()).split()).value_counts()

    df_top = series_top.to_frame(name="frequency")
    df_top['word'] = df_top.index
    df_top['word'] = df_top.word[df_top.word.str.len() > 2]
    df_top = df_top.dropna(axis=0)
    print(df_top)

    df_top = df_top.nlargest(6386, "frequency")
    print(df_top)

    df_top = df_top[['word', 'frequency']]

    # # # make csv file
    df_top.to_csv("simple.csv", index=False)


"""
For subtitles

1) Use collection dictionaries to map each word to a frequency
2) Turn this into a pandas dataframe
3) Keep the top 1000 word frequencies


"""


def subtitles():

    words = re.findall(
        '\w+', open('camb_model/binary-features/subtitles.txt').read().lower())

    word_dict = collections.Counter(words)
    df = pd.DataFrame.from_dict(
        word_dict, orient='index', columns=['frequency'])

    # filter and keep top 1000 word frequency
    df_top = df.nlargest(6386, 'frequency')
    print(df_top)

    # make csv file
    df_top.to_csv("subtitles.csv", index=True)


# simple_wiki()
# subtitles()
