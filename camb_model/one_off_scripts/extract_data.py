import pandas as pd

import re
import collections
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


def simple_wiki():
    colnames = ['word', 'paragraph', 'sentence']
    df_wiki = pd.read_table('camb_model/corpus/simple.txt',
                            names=colnames, header=None)

    series_top = pd.Series(
        " ".join((df_wiki.word).str.lower()).split()).value_counts()

    df_top = series_top.to_frame(name="frequency")
    print(df_top)

    df_top = df_top.nlargest(6386, "frequency")

    # # make csv file
    df_top.to_csv("simple.csv", index=True)


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


simple_wiki()
subtitles()
