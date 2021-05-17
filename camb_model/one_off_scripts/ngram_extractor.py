import nltk
import string
import pandas as pd
from nltk import tokenize
from nltk.util import ngrams
from collections import Counter

from nltk import ngrams
from pandas.io.sql import table_exists



# the real code goes here
#####################################

lgrams = []
n = 4

wiki_df = pd.read_csv("../corpus/german/wikipedia_corpus.csv")
# wiki_df = wiki_df[:1000]
tcount = 0
germ_chars = 'abcdefghijklmnopqrstuvwxyzäöüß'

for index, row in wiki_df.iterrows():
    
    bigrams = ngrams(row['word'],n)
    temp_grams = []

    try:
        [temp_grams.append(str(gram[0] + gram[1] + gram[2] + gram[3])) for gram in bigrams if (gram[0] in germ_chars and gram[1] in germ_chars and gram[2] in germ_chars and gram[3] in germ_chars)]
        temp_grams * row['frequency']
        lgrams += temp_grams
        tcount += len(temp_grams)
    except:
        continue

#####################################
# stick the counts into a dataframe

c = Counter(lgrams)
df = pd.DataFrame.from_dict(c, orient='index').reset_index()
df = df.rename(columns={'index':'fourgram', 0:'frequency'})

l_gramtypes = len(c)
df['probability'] = df['frequency'].apply(lambda x: x / tcount)
df = df.sort_values('frequency', ascending=False)

print(df)
df.to_csv("out.csv", index=False)